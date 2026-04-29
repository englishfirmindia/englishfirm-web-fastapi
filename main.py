from dotenv import load_dotenv
load_dotenv()

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

from routers.auth import router as auth_router
from routers.user import router as user_router
from routers.milestones import router as milestones_router
from routers.ai_assistant import router as ai_router
from routers.reports import router as reports_router
from routers.trainer.auth import router as trainer_auth_router
from routers.trainer.app import router as trainer_app_router
from routers.student_share import router as student_share_router

app = FastAPI(title="EnglishFirm Web API", version="1.0.0")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(
        "[UNHANDLED] %s %s — %s: %s",
        request.method, request.url.path, type(exc).__name__, exc,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

app.include_router(auth_router, prefix="/api/v1")
app.include_router(user_router, prefix="/api/v1")
app.include_router(milestones_router, prefix="/api/v1")
app.include_router(ai_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")
app.include_router(trainer_auth_router, prefix="/api/v1")
app.include_router(trainer_app_router, prefix="/api/v1")
app.include_router(student_share_router, prefix="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Speaking
from routers.speaking import read_aloud, repeat_sentence, answer_short_question
from routers.speaking import describe_image, retell_lecture, summarize_group_discussion, respond_to_situation
# Writing
from routers.writing import summarize_written_text, write_essay
# Reading
from routers.reading import fill_in_blanks, mcs as reading_mcs, mcm as reading_mcm
from routers.reading import reorder_paragraphs, fib_drag_drop
# Listening
from routers.listening import wfd, sst, fib as listening_fib
from routers.listening import mcs as listening_mcs, mcm as listening_mcm
from routers.listening import hcs, smw, hiw
# Sectional
from routers.sectional import speaking as sectional_speaking
from routers.sectional import writing as sectional_writing
from routers.sectional import reading as sectional_reading
from routers.sectional import listening as sectional_listening
# Mock (full PTE exam)
from routers.mock import router as mock_router
from routers.resources import router as resources_router
from routers.mic_check import router as mic_check_router

PREFIX = "/api/v1/questions"

for r in [
    read_aloud.router, repeat_sentence.router, answer_short_question.router,
    describe_image.router, retell_lecture.router, summarize_group_discussion.router,
    respond_to_situation.router,
    summarize_written_text.router, write_essay.router,
    fill_in_blanks.router, reading_mcs.router, reading_mcm.router,
    reorder_paragraphs.router, fib_drag_drop.router,
    wfd.router, sst.router, listening_fib.router, listening_mcs.router,
    listening_mcm.router, hcs.router, smw.router, hiw.router,
    sectional_speaking.router, sectional_writing.router,
    sectional_reading.router, sectional_listening.router,
    mock_router,
    resources_router,
    mic_check_router,
]:
    app.include_router(r, prefix=PREFIX)
