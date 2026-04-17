import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# S3 CLIENT
# Reads from .env locally, reads from IAM Role on EC2
# (boto3 automatically falls back to instance role when
#  env vars are not present — zero code change needed
#  when you deploy to EC2 with an attached IAM Role)
# ---------------------------------------------------------

def _get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_S3_REGION", "ap-southeast-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


# ---------------------------------------------------------
# GENERATE PRESIGNED URL
#
# Takes a full S3 URL like:
#   https://apeuni-questions-audio.s3.ap-southeast-2.amazonaws.com/rs/rs_1487.mp3
#
# Returns a presigned URL valid for 1 hour.
# Flutter plays this URL directly — no audio bytes
# go through your backend.
# ---------------------------------------------------------

PRESIGNED_URL_EXPIRY_SECONDS = 3600  # 1 hour


def generate_presigned_url(s3_url: str) -> str:
    """
    Converts a raw private S3 URL into a time-limited presigned URL.
    Raises ValueError if the URL is not a valid S3 URL.
    """

    # Parse bucket and key from URL
    # Input:  https://apeuni-questions-audio.s3.ap-southeast-2.amazonaws.com/rs/rs_1487.mp3
    # Bucket: apeuni-questions-audio
    # Key:    rs/rs_1487.mp3

    try:
        # Strip scheme
        without_scheme = s3_url.replace("https://", "").replace("http://", "")

        # Split host from path
        # without_scheme = "apeuni-questions-audio.s3.ap-southeast-2.amazonaws.com/rs/rs_1487.mp3"
        host, key = without_scheme.split("/", 1)

        # Extract bucket name (everything before the first dot)
        bucket = host.split(".")[0]

    except Exception:
        raise ValueError(f"Invalid S3 URL format: {s3_url}")

    if not bucket or not key:
        raise ValueError(f"Could not parse bucket/key from URL: {s3_url}")

    client = _get_s3_client()

    presigned_url = client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
        },
        ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
    )

    return presigned_url


UPLOAD_URL_EXPIRY_SECONDS = 300  # 5 minutes
_REGION = os.getenv("AWS_S3_REGION", "ap-southeast-2")


def _get_s3_upload_client():
    return boto3.client(
        "s3",
        region_name=_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def generate_presigned_upload_url(key: str) -> dict:
    """
    Generates a presigned PUT URL for uploading a user recording to S3.
    Returns { upload_url, s3_url }.
    """
    bucket = os.getenv("S3_RECORDINGS_BUCKET", "apeuni-user-recordings")
    client = _get_s3_upload_client()

    upload_url = client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ContentType": "audio/aac",
        },
        ExpiresIn=UPLOAD_URL_EXPIRY_SECONDS,
    )

    s3_url = f"https://{bucket}.s3.{_REGION}.amazonaws.com/{key}"
    return {"upload_url": upload_url, "s3_url": s3_url}
