from fastapi import APIRouter

router = APIRouter(prefix="/sectional/listening", tags=["Sectional - Listening"])


@router.get("/info")
def info():
    return {"status": "not_implemented"}


@router.post("/exam")
def start_exam():
    return {"status": "not_implemented"}


@router.post("/finish")
def finish_exam():
    return {"status": "not_implemented"}


@router.get("/results/{session_id}")
def get_results(session_id: str):
    return {"status": "not_implemented"}
