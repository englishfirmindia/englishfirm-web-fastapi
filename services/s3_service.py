import boto3
import os
from dotenv import load_dotenv

import core.config as config

load_dotenv()


def _get_s3_client():
    return boto3.client(
        "s3",
        region_name=config.AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def generate_presigned_url(s3_url: str) -> str:
    """
    Converts a raw private S3 URL into a time-limited presigned URL.
    Raises ValueError if the URL is not a valid S3 URL.
    """
    try:
        without_scheme = s3_url.replace("https://", "").replace("http://", "")
        host, key = without_scheme.split("/", 1)
        bucket = host.split(".")[0]
    except Exception:
        raise ValueError(f"Invalid S3 URL format: {s3_url}")

    if not bucket or not key:
        raise ValueError(f"Could not parse bucket/key from URL: {s3_url}")

    client = _get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=config.PRESIGNED_READ_EXPIRY_SECONDS,
    )


def _get_s3_upload_client():
    return boto3.client(
        "s3",
        region_name=config.AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def generate_presigned_upload_url(key: str) -> dict:
    """
    Generates a presigned PUT URL for uploading a user recording to S3.
    Returns { upload_url, s3_url }.
    """
    bucket = config.S3_RECORDINGS_BUCKET
    client = _get_s3_upload_client()

    upload_url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "ContentType": "audio/aac"},
        ExpiresIn=config.PRESIGNED_UPLOAD_EXPIRY_SECONDS,
    )

    s3_url = f"https://{bucket}.s3.{config.AWS_REGION}.amazonaws.com/{key}"
    return {"upload_url": upload_url, "s3_url": s3_url}
