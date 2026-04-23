"""
Static PTE learning resources — presigned URL generation.
All files live under s3://apeuni-questions-audio/resources/
"""
import os
import boto3
from botocore.config import Config

_BUCKET = "apeuni-questions-audio"
_REGION = os.getenv("AWS_S3_REGION", "ap-southeast-2")
_PREFIX = "resources"
_EXPIRY = 60 * 60 * 24  # 24 hours


def _s3():
    return boto3.client(
        "s3",
        region_name=_REGION,
        config=Config(signature_version="s3v4"),
    )


def _url(key: str) -> str:
    return _s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": _BUCKET, "Key": f"{_PREFIX}/{key}"},
        ExpiresIn=_EXPIRY,
    )


# ─── Resource catalogue ───────────────────────────────────────────────────────
# Each item: { title, key, type }
# type: pdf | pptx | docx | xlsx | png

_CATALOGUE = {
    "class_slides": {
        "label": "Class Slides",
        "icon": "slides",
        "color": 0xFF6366F1,  # indigo
        "subsections": {
            "speaking": {
                "label": "Speaking",
                "color": 0xFF8B5CF6,
                "items": [
                    {"title": "Read Aloud",                "key": "class-slides/speaking/Read Aloud.pptx",                    "type": "pptx"},
                    {"title": "Repeat Sentence",           "key": "class-slides/speaking/Repeat Sentence.pptx",               "type": "pptx"},
                    {"title": "Describe Image",            "key": "class-slides/speaking/Describe Image.pptx",                "type": "pptx"},
                    {"title": "Answer Short Question",     "key": "class-slides/speaking/Answer Short Question.pptx",          "type": "pptx"},
                    {"title": "Retell Lecture",            "key": "class-slides/speaking/Retell Lecture.pptx",                "type": "pptx"},
                    {"title": "Summarize Group Discussion","key": "class-slides/speaking/Summarize Group Discussion.pptx",     "type": "pptx"},
                    {"title": "Respond to a Situation",   "key": "class-slides/speaking/Respond to a Situation.pptx",        "type": "pptx"},
                ],
            },
            "writing": {
                "label": "Writing",
                "color": 0xFF3B82F6,
                "items": [
                    {"title": "Summarise Written Text", "key": "class-slides/writing/Summarise Written Text.pptx", "type": "pptx"},
                    {"title": "Write Essay",            "key": "class-slides/writing/Write Essay.pptx",            "type": "pptx"},
                ],
            },
            "reading": {
                "label": "Reading",
                "color": 0xFF10B981,
                "items": [
                    {"title": "MCQ Multiple Answers",           "key": "class-slides/reading/MCQ Multiple Answers.pptx",             "type": "pptx"},
                    {"title": "MCQ Single Answer",              "key": "class-slides/reading/MCQ Single Answer.pptx",                "type": "pptx"},
                    {"title": "Reorder Paragraphs",             "key": "class-slides/reading/Reorder Paragraphs.pptx",               "type": "pptx"},
                    {"title": "Fill in the Blanks (Drag Drop)", "key": "class-slides/reading/Fill in the Blanks (Drag Drop).pptx",   "type": "pptx"},
                    {"title": "Fill in the Blanks (Drop Down)", "key": "class-slides/reading/Fill in the Blanks (Drop Down).pptx",   "type": "pptx"},
                ],
            },
            "listening": {
                "label": "Listening",
                "color": 0xFFF59E0B,
                "items": [
                    {"title": "Summarise Spoken Text",   "key": "class-slides/listening/Summarise Spoken Text.pptx",    "type": "pptx"},
                    {"title": "Fill in the Blanks",      "key": "class-slides/listening/Fill in the Blanks.pptx",       "type": "pptx"},
                    {"title": "MCQ Multiple Answers",    "key": "class-slides/listening/MCQ Multiple Answers.pptx",     "type": "pptx"},
                    {"title": "MCQ Single Answer",       "key": "class-slides/listening/MCQ Single Answer.pptx",        "type": "pptx"},
                    {"title": "Select Missing Word",     "key": "class-slides/listening/Select Missing Word.pptx",      "type": "pptx"},
                    {"title": "Highlight Correct Summary","key": "class-slides/listening/Highlight Correct Summary.pptx","type": "pptx"},
                    {"title": "Highlight Incorrect Words","key": "class-slides/listening/Highlight Incorrect Words.pptx","type": "pptx"},
                    {"title": "Write from Dictation",    "key": "class-slides/listening/Write from Dictation.pptx",     "type": "pptx"},
                ],
            },
            "updates": {
                "label": "Latest Updates",
                "color": 0xFFEF4444,
                "items": [
                    {"title": "New Updates in PTE Aug 2025", "key": "class-slides/New Updates in PTE Aug 2025.docx", "type": "docx"},
                ],
            },
        },
    },

    "templates": {
        "label": "Templates",
        "icon": "template",
        "color": 0xFF0EA5E9,  # sky
        "subsections": {
            "speaking": {
                "label": "Speaking",
                "color": 0xFF8B5CF6,
                "items": [
                    {"title": "Describe Image Template",          "key": "templates/speaking/Describe Image Template.docx",                 "type": "docx"},
                    {"title": "Retell Lecture Template",          "key": "templates/speaking/Retell Lecture Template.docx",                 "type": "docx"},
                    {"title": "Summarise Group Discussion",       "key": "templates/speaking/Summarise Group Discussion Template.docx",     "type": "docx"},
                    {"title": "Respond to a Situation",          "key": "templates/speaking/Respond to a Situation Template.docx",         "type": "docx"},
                ],
            },
            "writing": {
                "label": "Writing",
                "color": 0xFF3B82F6,
                "items": [
                    {"title": "Essay Template",                  "key": "templates/writing/Essay Template.docx",               "type": "docx"},
                    {"title": "Essay Template Examples",         "key": "templates/writing/Essay Template Examples.docx",       "type": "docx"},
                    {"title": "Most Repeated Essay Questions",   "key": "templates/writing/Most Repeated Essay Questions.docx", "type": "docx"},
                ],
            },
            "listening": {
                "label": "Listening",
                "color": 0xFFF59E0B,
                "items": [
                    {"title": "SST Template",           "key": "templates/listening/SST Template.docx",          "type": "docx"},
                    {"title": "WFD Full Word List",     "key": "templates/listening/WFD Full Word List.docx",    "type": "docx"},
                    {"title": "Exhaustive SST List",    "key": "templates/listening/Exhaustive SST List.docx",   "type": "docx"},
                ],
            },
            "general": {
                "label": "General",
                "color": 0xFF6B7280,
                "items": [
                    {"title": "All Templates for Print", "key": "templates/general/Templates for Print.docx", "type": "docx"},
                ],
            },
        },
    },

    "vocab_grammar": {
        "label": "Vocabulary & Grammar",
        "icon": "vocab",
        "color": 0xFF059669,  # emerald
        "subsections": {
            "all": {
                "label": "All Resources",
                "color": 0xFF059669,
                "items": [
                    {"title": "Phrasal Verbs",               "key": "vocab-grammar/Phrasal Verbs.pdf",                "type": "pdf"},
                    {"title": "IDIOMS",                      "key": "vocab-grammar/IDIOMS.docx",                      "type": "docx"},
                    {"title": "Transition & Linking Words",  "key": "vocab-grammar/Transition and Linking Words.docx", "type": "docx"},
                    {"title": "Phrases and Useful Words",    "key": "vocab-grammar/Phrases and Useful Words.docx",     "type": "docx"},
                    {"title": "Grammatical Cues",            "key": "vocab-grammar/Grammatical Cues.docx",             "type": "docx"},
                    {"title": "Parts of Speech",             "key": "vocab-grammar/Parts of Speech.pptx",              "type": "pptx"},
                    {"title": "1K Academic Words",           "key": "vocab-grammar/1K Academic Words.docx",            "type": "docx"},
                    {"title": "PTE Collocation List",        "key": "vocab-grammar/PTE Collocation List.pdf",          "type": "pdf"},
                    {"title": "Grammar and Spelling",        "key": "vocab-grammar/Grammar and Spelling.pdf",          "type": "pdf"},
                    {"title": "English Grammar in Use",      "key": "vocab-grammar/English Grammar in Use.pdf",        "type": "pdf"},
                ],
            },
        },
    },

    "spelling": {
        "label": "Spelling Rules",
        "icon": "spelling",
        "color": 0xFFF97316,  # orange
        "subsections": {
            "all": {
                "label": "All Resources",
                "color": 0xFFF97316,
                "items": [
                    {"title": "Spelling Rules",             "key": "spelling/Spelling Rules.pdf",                "type": "pdf"},
                    {"title": "Advanced Spelling Rules",    "key": "spelling/Advanced Spelling Rules.pdf",       "type": "pdf"},
                    {"title": "More Links to Learn Spelling","key": "spelling/More Links to Learn Spelling.docx","type": "docx"},
                ],
            },
        },
    },

    "official_guides": {
        "label": "PTE Official Guides",
        "icon": "guide",
        "color": 0xFFDC2626,  # red
        "subsections": {
            "all": {
                "label": "All Resources",
                "color": 0xFFDC2626,
                "items": [
                    {"title": "PTE Score Guide Jul 2025",         "key": "official-guides/PTE Score Guide Jul 2025.pdf", "type": "pdf"},
                    {"title": "PTE Scoring Info for Partners",    "key": "official-guides/PTE Scoring Info.pdf",         "type": "pdf"},
                    {"title": "PTE Score Distribution Chart",     "key": "quick-ref/PTE Score Distribution Chart.png",   "type": "png"},
                    {"title": "Common PTE Questions & Answers",   "key": "official-guides/Common PTE Questions and Answers.docx", "type": "docx"},
                ],
            },
        },
    },

    "quick_ref": {
        "label": "Quick Reference",
        "icon": "quickref",
        "color": 0xFF7C3AED,  # violet
        "subsections": {
            "all": {
                "label": "All Resources",
                "color": 0xFF7C3AED,
                "items": [
                    {"title": "PTE vs IELTS Comparison",       "key": "quick-ref/PTE vs IELTS Comparison.png",        "type": "png"},
                    {"title": "ASQ Repeated Questions",        "key": "quick-ref/ASQ Repeated Questions.docx",        "type": "docx"},
                    {"title": "Most Repeated Essay Questions", "key": "quick-ref/Most Repeated Essay Questions.docx", "type": "docx"},
                    {"title": "PTE Practice Schedule",         "key": "quick-ref/PTE Practice Schedule.xlsx",         "type": "xlsx"},
                    {"title": "Online Class Timetable",        "key": "quick-ref/Online Class Timetable.docx",        "type": "docx"},
                    {"title": "Face to Face Class Timetable",  "key": "quick-ref/Face to Face Class Timetable.docx",  "type": "docx"},
                    {"title": "Class Videos Links",            "key": "quick-ref/Class Videos Links.docx",            "type": "docx"},
                ],
            },
        },
    },
}


def get_resources() -> list:
    """
    Returns all categories with presigned URLs injected into each item.
    Structure:
      [ { id, label, icon, color, subsections: [ { label, color, items: [ {title, url, type} ] } ] } ]
    """
    result = []
    for cat_id, cat in _CATALOGUE.items():
        subsections = []
        for sub_id, sub in cat["subsections"].items():
            items = []
            for item in sub["items"]:
                try:
                    url = _url(item["key"])
                except Exception:
                    url = ""
                items.append({
                    "title": item["title"],
                    "url":   url,
                    "type":  item["type"],
                })
            subsections.append({
                "id":    sub_id,
                "label": sub["label"],
                "color": sub["color"],
                "items": items,
            })
        result.append({
            "id":          cat_id,
            "label":       cat["label"],
            "icon":        cat["icon"],
            "color":       cat["color"],
            "subsections": subsections,
        })
    return result
