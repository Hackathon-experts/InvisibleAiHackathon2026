# aitools.py
"""
AI tools for processing lecture transcripts using the DeepSeek API.

Public functions:
    - summarise_transcript(transcript, info=None) -> dict
    - create_quiz(notes, num_questions, info=None) -> dict
    - mark_answer(question, user_answer, marking_guide, max_marks, info=None) -> dict
    - make_flashcards(transcript, info=None) -> dict

All functions return plain dicts/lists so they can be serialised to JSON
and passed between services (e.g. a web API) without modification.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

_DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
_DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
_MAX_RETRIES = 3
_TEMPERATURE = 0.2  # low temperature for consistency across calls

_client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=_DEEPSEEK_BASE_URL,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

class AIOutputError(RuntimeError):
    """Raised when the model fails to produce valid output after retries."""


def _call_model(system_prompt: str, user_prompt: str) -> str:
    """Single raw call to DeepSeek. Returns the text content."""
    response = _client.chat.completions.create(
        model=_DEEPSEEK_MODEL,
        temperature=_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _extract_json(text: str) -> Any:
    """
    Best-effort extraction of a JSON object/array from model output.
    Handles ```json fences and stray prose around the JSON.
    """
    text = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the outermost {...} or [...]
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise json.JSONDecodeError("No JSON found in response", text, 0)


def _call_json(
    system_prompt: str,
    user_prompt: str,
    validator,
    max_retries: int = _MAX_RETRIES,
) -> Any:
    """
    Call the model and parse JSON with retries.

    validator: callable(parsed) -> None. Raise ValueError on bad shape.
    On failure, the error is fed back to the model so it can correct itself.
    """
    last_error: Exception | None = None
    convo_suffix = ""

    for attempt in range(1, max_retries + 1):
        raw = _call_model(system_prompt, user_prompt + convo_suffix)
        try:
            parsed = _extract_json(raw)
            validator(parsed)
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            convo_suffix = (
                f"\n\nYour previous response was invalid: {exc}.\n"
                "Return ONLY valid JSON matching the required schema. "
                "No prose, no markdown fences, no explanations."
            )

    raise AIOutputError(
        f"Model failed to produce valid output after {max_retries} attempts: {last_error}"
    )


def _info_line(info: dict | None) -> str:
    """Format the optional `info` dict into a prompt line."""
    if not info:
        return ""
    parts = [f"{k}: {v}" for k, v in info.items() if v]
    return "\nLearner context: " + "; ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# 1. summarise_transcript
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = """You are an academic note-taker. You convert raw lecture \
transcripts into clean, factual study notes.

Rules:
- Stick strictly to what the transcript says. Do NOT add outside knowledge, \
examples, opinions, or filler.
- Remove all informalities: greetings, jokes, "um/uh", off-topic chat, \
audience banter, admin announcements, and repetition.
- Keep only content useful for study: concepts, definitions, explanations, \
processes, formulas, and worked reasoning.
- Preserve technical terms and the lecturer's terminology exactly.
- Organise the notes with short headings and bullet points.
- Do NOT invent section titles that aren't supported by the content.

You MUST respond with a single JSON object with this exact schema:
{
  "title": "<short descriptive title, or 'Lecture Notes' if unclear>",
  "summary": "<2-4 sentence overview>",
  "sections": [
    {"heading": "<heading>", "points": ["<point>", "<point>"]}
  ],
  "key_terms": ["<term>", "<term>"]
}
Return ONLY the JSON. No markdown fences. No commentary.
"""


def _validate_summary(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Top-level output must be a JSON object.")
    for key in ("title", "summary", "sections", "key_terms"):
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    if not isinstance(data["sections"], list):
        raise ValueError("'sections' must be a list.")
    for i, sec in enumerate(data["sections"]):
        if not isinstance(sec, dict):
            raise ValueError(f"sections[{i}] must be an object.")
        if "heading" not in sec or "points" not in sec:
            raise ValueError(f"sections[{i}] must have 'heading' and 'points'.")
        if not isinstance(sec["points"], list) or not all(
            isinstance(p, str) for p in sec["points"]
        ):
            raise ValueError(f"sections[{i}].points must be a list of strings.")
    if not isinstance(data["key_terms"], list) or not all(
        isinstance(t, str) for t in data["key_terms"]
    ):
        raise ValueError("'key_terms' must be a list of strings.")


def summarise_transcript(transcript: str, info: dict | None = None) -> dict:
    """
    Summarise a lecture transcript into structured study notes.

    Parameters
    ----------
    transcript : str
        Raw transcript text.
    info : dict, optional
        Extra learner context, e.g. {"level": "undergraduate year 2",
        "subject": "organic chemistry"}.

    Returns
    -------
    dict with schema:
        {
            "title": str,
            "summary": str,
            "sections": [
                {"heading": str, "points": [str, ...]},
                ...
            ],
            "key_terms": [str, ...]
        }
    """
    if not transcript or not transcript.strip():
        raise ValueError("transcript must be a non-empty string.")

    user_prompt = (
        f"Summarise the following lecture transcript into study notes."
        f"{_info_line(info)}\n\n"
        "--- TRANSCRIPT START ---\n"
        f"{transcript}\n"
        "--- TRANSCRIPT END ---"
    )
    return _call_json(_SUMMARY_SYSTEM, user_prompt, _validate_summary)


# ---------------------------------------------------------------------------
# 2. create_quiz
# ---------------------------------------------------------------------------

_QUIZ_SYSTEM = """You are an exam writer. You create quizzes from lecture notes.

Rules:
- Base every question strictly on the provided notes. Do NOT use outside knowledge.
- Produce exactly the requested number of questions.
- Mix question types: roughly 60% multiple choice, 40% short text response.
  Adjust if the requested number is small.
- Each question must have a mark value reflecting its difficulty (1-5 marks).
- Multiple choice questions: provide exactly 4 options labelled "A"-"D", \
exactly one correct option, and a short marking guide explaining the correct \
answer.
- Text response questions: provide a model answer and a short marking guide \
listing what earns marks.
- Keep wording clear, unambiguous, and consistent in style.
- Do NOT repeat questions or cover the same point twice.

You MUST respond with a single JSON object with this exact schema:
{
  "quiz_title": "<title>",
  "total_marks": <int>,
  "questions": [
    {
      "id": 1,
      "type": "multiple_choice",
      "question": "<question text>",
      "options": {
        "A": "<option>",
        "B": "<option>",
        "C": "<option>",
        "D": "<option>"
      },
      "correct_option": "<A|B|C|D>",
      "marking_guide": "<why the correct option is correct>",
      "marks": <int>
    },
    {
      "id": 2,
      "type": "text_response",
      "question": "<question text>",
      "model_answer": "<full model answer>",
      "marking_guide": "<bullet-style criteria for awarding marks>",
      "marks": <int>
    }
  ]
}
Return ONLY the JSON. No markdown fences. No commentary.
"""


def _validate_quiz(data: Any, expected_n: int) -> None:
    if not isinstance(data, dict):
        raise ValueError("Top-level output must be a JSON object.")
    for key in ("quiz_title", "total_marks", "questions"):
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    if not isinstance(data["questions"], list):
        raise ValueError("'questions' must be a list.")
    if len(data["questions"]) != expected_n:
        raise ValueError(
            f"Expected {expected_n} questions, got {len(data['questions'])}."
        )

    running_total = 0
    for i, q in enumerate(data["questions"]):
        if not isinstance(q, dict):
            raise ValueError(f"questions[{i}] must be an object.")
        for key in ("id", "type", "question", "marks"):
            if key not in q:
                raise ValueError(f"questions[{i}] missing '{key}'.")
        if q["type"] not in ("multiple_choice", "text_response"):
            raise ValueError(
                f"questions[{i}].type must be 'multiple_choice' or 'text_response'."
            )
        if not isinstance(q["marks"], int) or q["marks"] <= 0:
            raise ValueError(f"questions[{i}].marks must be a positive int.")

        if q["type"] == "multiple_choice":
            for key in ("options", "correct_option", "marking_guide"):
                if key not in q:
                    raise ValueError(f"MC question {i} missing '{key}'.")
            if not isinstance(q["options"], dict) or set(q["options"].keys()) != {
                "A", "B", "C", "D",
            }:
                raise ValueError(
                    f"MC question {i} 'options' must have exactly A, B, C, D."
                )
            if q["correct_option"] not in ("A", "B", "C", "D"):
                raise ValueError(
                    f"MC question {i} 'correct_option' must be A/B/C/D."
                )
        else:
            for key in ("model_answer", "marking_guide"):
                if key not in q:
                    raise ValueError(f"Text question {i} missing '{key}'.")

        running_total += q["marks"]

    if data["total_marks"] != running_total:
        raise ValueError(
            f"'total_marks' ({data['total_marks']}) does not match sum of "
            f"question marks ({running_total})."
        )


def create_quiz(notes: str | dict, num_questions: int, info: dict | None = None) -> dict:
    """
    Create a mixed multiple-choice / text-response quiz from lecture notes.

    Parameters
    ----------
    notes : str | dict
        The lecture notes. Pass the output of `summarise_transcript` directly,
        or a plain string.
    num_questions : int
        Exact number of questions to generate (> 0).
    info : dict, optional
        Extra learner context (e.g. level of study, subject).

    Returns
    -------
    dict with schema:
        {
            "quiz_title": str,
            "total_marks": int,
            "questions": [
                # multiple choice
                {
                    "id": int,
                    "type": "multiple_choice",
                    "question": str,
                    "options": {"A": str, "B": str, "C": str, "D": str},
                    "correct_option": "A"|"B"|"C"|"D",
                    "marking_guide": str,
                    "marks": int
                },
                # text response
                {
                    "id": int,
                    "type": "text_response",
                    "question": str,
                    "model_answer": str,
                    "marking_guide": str,
                    "marks": int
                }
            ]
        }
    """
    if not isinstance(num_questions, int) or num_questions <= 0:
        raise ValueError("num_questions must be a positive integer.")

    notes_text = (
        json.dumps(notes, ensure_ascii=False, indent=2)
        if isinstance(notes, dict)
        else str(notes)
    )

    user_prompt = (
        f"Create a quiz with EXACTLY {num_questions} questions from the notes below."
        f"{_info_line(info)}\n\n"
        "--- NOTES START ---\n"
        f"{notes_text}\n"
        "--- NOTES END ---"
    )

    return _call_json(
        _QUIZ_SYSTEM,
        user_prompt,
        lambda data: _validate_quiz(data, num_questions),
    )


# ---------------------------------------------------------------------------
# 3. mark_answer
# ---------------------------------------------------------------------------

_MARK_SYSTEM = """You are a strict but fair examiner. You mark a single student \
answer against a marking guide and a maximum mark.

Rules:
- Award marks strictly according to the marking guide.
- Do NOT award marks for content not present in the student's answer.
- If the answer is empty, irrelevant, or off-topic, award 0.
- The awarded mark must be an integer between 0 and the maximum mark (inclusive).
- Keep feedback concise (1-3 sentences), specific to the student's answer.
- Be consistent: identical answers to identical questions get identical marks.

You MUST respond with a single JSON object with this exact schema:
{
  "mark": <int>,
  "max_marks": <int>,
  "feedback": "<short feedback>",
  "criteria_met": ["<criterion>", ...],
  "criteria_missed": ["<criterion>", ...]
}
Return ONLY the JSON. No markdown fences. No commentary.
"""


def _validate_mark(data: Any, max_marks: int) -> None:
    if not isinstance(data, dict):
        raise ValueError("Top-level output must be a JSON object.")
    for key in ("mark", "max_marks", "feedback"):
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    if not isinstance(data["mark"], int):
        raise ValueError("'mark' must be an integer.")
    if data["mark"] < 0 or data["mark"] > max_marks:
        raise ValueError(
            f"'mark' ({data['mark']}) must be between 0 and {max_marks}."
        )
    if data["max_marks"] != max_marks:
        raise ValueError(
            f"'max_marks' ({data['max_marks']}) must equal {max_marks}."
        )
    if not isinstance(data["feedback"], str):
        raise ValueError("'feedback' must be a string.")
    # Optional fields
    for key in ("criteria_met", "criteria_missed"):
        if key in data and not isinstance(data[key], list):
            raise ValueError(f"'{key}' must be a list if present.")


def mark_answer(
    question: str,
    user_answer: str,
    marking_guide: str,
    max_marks: int,
    info: dict | None = None,
) -> dict:
    """
    Mark a single student answer.

    Parameters
    ----------
    question : str
        The question text.
    user_answer : str
        The student's answer.
    marking_guide : str
        Marking guide produced by `create_quiz`.
    max_marks : int
        Maximum marks available for this question.
    info : dict, optional
        Extra learner context.

    Returns
    -------
    dict with schema:
        {
            "mark": int,                 # 0..max_marks
            "max_marks": int,            # == max_marks
            "feedback": str,
            "criteria_met": [str, ...],
            "criteria_missed": [str, ...]
        }
    """
    if not isinstance(max_marks, int) or max_marks <= 0:
        raise ValueError("max_marks must be a positive integer.")

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Marking guide:\n{marking_guide}\n\n"
        f"Maximum marks: {max_marks}\n"
        f"{_info_line(info)}\n\n"
        f"Student answer:\n{user_answer if user_answer else '(blank)'}"
    )

    return _call_json(
        _MARK_SYSTEM,
        user_prompt,
        lambda data: _validate_mark(data, max_marks),
    )


# ---------------------------------------------------------------------------
# 4. make_flashcards
# ---------------------------------------------------------------------------

_FLASHCARD_SYSTEM = """You extract definitions and term/meaning pairs from a \
lecture transcript to build flashcards.

Rules:
- Only include terms that the transcript explicitly defines or explains.
- Do NOT add outside knowledge, examples, or your own definitions.
- If the lecturer uses a phrase like "X is ...", "X refers to ...", \
"X means ...", "by X we mean ...", extract that.
- Keep definitions concise (1-2 sentences) and faithful to the transcript.
- Use the exact term the lecturer uses.
- Skip trivial or non-academic items (e.g. names of people, admin items).

You MUST respond with a single JSON object with this exact schema:
{
  "flashcards": [
    {"term": "<term>", "definition": "<definition>"}
  ]
}
Return ONLY the JSON. No markdown fences. No commentary.
"""


def _validate_flashcards(data: Any) -> None:
    if not isinstance(data, dict) or "flashcards" not in data:
        raise ValueError("Top-level output must be an object with 'flashcards'.")
    cards = data["flashcards"]
    if not isinstance(cards, list):
        raise ValueError("'flashcards' must be a list.")
    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            raise ValueError(f"flashcards[{i}] must be an object.")
        if "term" not in card or "definition" not in card:
            raise ValueError(f"flashcards[{i}] must have 'term' and 'definition'.")
        if not isinstance(card["term"], str) or not isinstance(
            card["definition"], str
        ):
            raise ValueError(f"flashcards[{i}] values must be strings.")


def make_flashcards(transcript: str, info: dict | None = None) -> dict:
    """
    Extract term/definition pairs from a lecture transcript.

    Parameters
    ----------
    transcript : str
        Raw transcript text.
    info : dict, optional
        Extra learner context.

    Returns
    -------
    dict with schema:
        {
            "flashcards": [
                {"term": str, "definition": str},
                ...
            ]
        }
    """
    if not transcript or not transcript.strip():
        raise ValueError("transcript must be a non-empty string.")

    user_prompt = (
        "Extract flashcards from the following lecture transcript."
        f"{_info_line(info)}\n\n"
        "--- TRANSCRIPT START ---\n"
        f"{transcript}\n"
        "--- TRANSCRIPT END ---"
    )
    return _call_json(_FLASHCARD_SYSTEM, user_prompt, _validate_flashcards)