"""Phase 7: feedback for the student on each answer.

* template_feedback(): always available, free, instant. Built from the missed / half points.
  Filled in automatically when a script is graded.
* llm_feedback(): optional. A LangChain chain (prompt template -> Cohere chat model ->
  Pydantic output parser) that turns the same information into a short, friendly comment.
  Needs COHERE_API_KEY in the .env file. Only the question, marking points and answer are
  sent - never the student's name or roll number.
The teacher can edit either version before sharing it.
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _get(obj, key):
    """Points may be dataclasses (during grading) or dicts (from the database)."""
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


# ---------------------------------------------------------------- template feedback
def template_feedback(method: str, points: list, suggested: float | None, max_marks: float) -> str:
    if method == "missing":
        return "This question was not answered."
    if method == "mcq":
        return "" if suggested == max_marks else "The option chosen is not correct."
    if method != "pointwise" or not points:
        return ""
    missed = [_get(p, "text") for p in points if _get(p, "status") == "missed"]
    half = [_get(p, "text") for p in points if _get(p, "status") == "half"]
    if not missed and not half:
        return "Good answer: all the key points are covered."
    parts = []
    if missed:
        parts.append("Missing: " + "; ".join(missed) + ".")
    if half:
        parts.append("Explain more clearly: " + "; ".join(half) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------- LLM feedback (LangChain)
class AnswerFeedback(BaseModel):
    """The structure we ask the LLM to return (lab 9: Pydantic schema + output parser)."""
    strength: str = Field(description="One short sentence on what the student did well. "
                                      "Empty string if nothing in the answer was correct.")
    missing: list[str] = Field(description="Up to 3 short phrases naming what was missing or wrong, "
                                           "in simple words a school student understands.")
    advice: str = Field(description="One short sentence on how to write a better answer next time.")

    def as_text(self) -> str:
        parts = []
        if self.strength.strip():
            parts.append(self.strength.strip())
        if self.missing:
            parts.append("To improve: " + "; ".join(m.strip().rstrip(".") for m in self.missing[:3]) + ".")
        if self.advice.strip():
            parts.append(self.advice.strip())
        return " ".join(parts)


SYSTEM_PROMPT = (
    "You are an experienced CBSE school teacher writing short, kind and specific feedback on a "
    "student's exam answer. Use simple English. Base the feedback ONLY on the marking points "
    "given - do not invent new requirements. Do not mention marks, scores, AI or similarity. "
    "Address the student as 'you'.\n\n{format_instructions}"
)
HUMAN_PROMPT = (
    "Question: {question}\n"
    "Marking points the answer covered: {covered}\n"
    "Marking points missed or only partly covered: {missed}\n"
    "Student's answer: {answer}\n\n"
    "Write the feedback."
)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(config.ROOT / ".env")
    except ImportError:
        pass


def llm_available() -> bool:
    _load_env()
    return bool(os.environ.get("COHERE_API_KEY"))


@lru_cache(maxsize=1)
def _chain():
    """prompt | model | parser - built once."""
    from langchain_cohere import ChatCohere
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    _load_env()
    parser = PydanticOutputParser(pydantic_object=AnswerFeedback)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())
    model = ChatCohere(model=config.FEEDBACK_MODEL, temperature=0.2,
                       cohere_api_key=os.environ["COHERE_API_KEY"])
    return prompt | model | parser


def llm_feedback(question: str, answer: str, points: list) -> str:
    covered = [_get(p, "text") for p in points if _get(p, "status") == "full"]
    missed = [_get(p, "text") for p in points if _get(p, "status") != "full"]
    result: AnswerFeedback = _chain().invoke({
        "question": question,
        "covered": "; ".join(covered) or "none",
        "missed": "; ".join(missed) or "none",
        "answer": answer,
    })
    return result.as_text()