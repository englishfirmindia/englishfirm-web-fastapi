from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.auth import router as auth_router
from routers.user import router as user_router

app = FastAPI(title="EnglishFirm Web API", version="1.0.0")

app.include_router(auth_router, prefix="/api/v1")
app.include_router(user_router, prefix="/api/v1")

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
]:
    app.include_router(r, prefix=PREFIX)
