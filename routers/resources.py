from fastapi import APIRouter, Depends
from db.models import User
from core.dependencies import get_current_user
from services.billing.require_paid_plan import RequireLearnAccess

router = APIRouter(tags=["Resources"])

# Both endpoints accept EITHER an active paid plan OR a user with the
# perpetual `unlimited_learn_access` flag (set by trainer-granted VIPs
# with the lifetime-Learn checkbox on). Free users without the flag
# hit a 402 PLAN_LIMIT_REACHED with feature_key='resources', which the
# Flutter ApiClient turns into the existing upgrade sheet.
_learn_gate = RequireLearnAccess()


@router.get("/resources")
def get_learning_resources(
    current_user: User = Depends(get_current_user),
    _gate=Depends(_learn_gate),
):
    """Returns all PTE learning resources with 24-hour presigned S3 URLs."""
    from services.resources_service import get_resources
    return get_resources()


@router.get("/resources/file")
def get_resource_file(
    key: str,
    current_user: User = Depends(get_current_user),
    _gate=Depends(_learn_gate),
):
    """Returns a presigned URL for a single resource file."""
    from services.resources_service import _url
    import urllib.parse
    return {"url": _url(urllib.parse.unquote(key))}
