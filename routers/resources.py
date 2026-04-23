from fastapi import APIRouter, Depends
from db.models import User
from core.dependencies import get_current_user

router = APIRouter(prefix="/questions", tags=["Resources"])


@router.get("/resources")
def get_learning_resources(current_user: User = Depends(get_current_user)):
    """Returns all PTE learning resources with 24-hour presigned S3 URLs."""
    from services.resources_service import get_resources
    return get_resources()


@router.get("/resources/file")
def get_resource_file(key: str, current_user: User = Depends(get_current_user)):
    """Returns a presigned URL for a single resource file."""
    from services.resources_service import _url
    import urllib.parse
    return {"url": _url(urllib.parse.unquote(key))}
