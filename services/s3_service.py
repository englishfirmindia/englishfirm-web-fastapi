import boto3
import os
import time
from typing import Optional, Callable, TypeVar
from dotenv import load_dotenv
from botocore.config import Config as _BotoConfig
from botocore.exceptions import (
    EndpointConnectionError,
    ConnectionError as _BotoConnectionError,
    ReadTimeoutError,
    NoCredentialsError,
    PartialCredentialsError,
    ClientError,
)

import core.config as config
from core.logging_config import get_logger

load_dotenv()

log = get_logger(__name__)

# W10: tighten boto3 client timeouts — these helpers sign URLs locally so the
# only outbound traffic is occasional credential refresh from STS / IMDS.
# Default 60s connect+read is way too lax for what should be near-instant ops.
_S3_BOTO_CONFIG = _BotoConfig(
    connect_timeout=5,
    read_timeout=5,
    retries={"max_attempts": 1, "mode": "standard"},  # we wrap our own retry
)

# Module-level singleton — boto3 clients are thread-safe and designed to be
# reused. Re-creating per request causes a fresh TLS handshake every time.
_S3_CLIENT = boto3.client(
    "s3",
    region_name=config.AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    config=_S3_BOTO_CONFIG,
)

T = TypeVar("T")

# W3 / W4: errors worth retrying. Excludes credential / permission errors —
# those are config bugs that retries can't fix and we don't want to mask.
_RECOVERABLE_BOTO_EXCEPTIONS = (
    EndpointConnectionError,
    _BotoConnectionError,
    ReadTimeoutError,
)


def _with_boto_retry(label: str, fn: Callable[[], T]) -> T:
    """Run a boto3 call with 2 short retries on transient network errors.

    Backoff is 200ms then 500ms — kept tight because these helpers sit on
    the synchronous request path. Config errors (NoCredentials, AccessDenied)
    bubble immediately so we don't hide misconfiguration behind sleeps.
    """
    last_exc: Exception = RuntimeError(f"{label}: no attempts made")
    delays = (0.2, 0.5)
    for attempt in range(1, 4):
        try:
            return fn()
        except _RECOVERABLE_BOTO_EXCEPTIONS as exc:
            last_exc = exc
            log.warning("[%s] transient boto error attempt=%d/3: %s", label, attempt, exc)
            if attempt < 3:
                time.sleep(delays[attempt - 1])
        except (NoCredentialsError, PartialCredentialsError) as exc:
            log.error("[%s] credential error (no retry): %s", label, exc)
            raise
        except ClientError as exc:
            # AccessDenied / NoSuchKey / etc. — config or data issue, no retry.
            log.error("[%s] boto ClientError (no retry): %s", label, exc)
            raise
    log.error("[%s] failed after 3 attempts: %s", label, last_exc)
    raise last_exc


def generate_presigned_url(s3_url: str, expires_in: Optional[int] = None) -> str:
    """
    Converts a raw private S3 URL into a time-limited presigned URL.
    Raises ValueError if the URL is not a valid S3 URL.

    `expires_in` overrides the default PRESIGNED_READ_EXPIRY_SECONDS — used
    for trainer-side audio review where we want a longer-lived URL.
    """
    try:
        without_scheme = s3_url.replace("https://", "").replace("http://", "")
        host, key = without_scheme.split("/", 1)
        bucket = host.split(".")[0]
    except Exception:
        raise ValueError(f"Invalid S3 URL format: {s3_url}")

    if not bucket or not key:
        raise ValueError(f"Could not parse bucket/key from URL: {s3_url}")

    ttl = expires_in if expires_in is not None else config.PRESIGNED_READ_EXPIRY_SECONDS
    return _with_boto_retry(
        "S3_PRESIGN_GET",
        lambda: _S3_CLIENT.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl,
        ),
    )


def generate_presigned_upload_url(key: str) -> dict:
    """
    Generates a presigned PUT URL for uploading a user recording to S3.
    Returns { upload_url, s3_url }.
    """
    bucket = config.S3_RECORDINGS_BUCKET
    upload_url = _with_boto_retry(
        "S3_PRESIGN_PUT",
        lambda: _S3_CLIENT.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": key, "ContentType": "audio/aac"},
            ExpiresIn=config.PRESIGNED_UPLOAD_EXPIRY_SECONDS,
        ),
    )

    s3_url = f"https://{bucket}.s3.{config.AWS_REGION}.amazonaws.com/{key}"
    return {"upload_url": upload_url, "s3_url": s3_url}
