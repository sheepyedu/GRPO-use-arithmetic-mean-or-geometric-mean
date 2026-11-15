"""Strict final-answer parsers. Model text is never evaluated as code."""

import re
from fractions import Fraction

NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+)?")


def number(text):
    value = text.strip()
    if not NUMBER.fullmatch(value):
        return None
    try:
        return Fraction(value.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def parse_answer(text, task, labels=None, truncated=False):
    # Thinking text may discuss the required marker. Only the post-thinking final
    # channel is eligible, and it must contain exactly one complete answer line.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None
    if text.count("Final answer:") != 1:
        return None
    prefix, answer = text.split("Final answer:")
    if truncated and not answer.endswith("\n"):
        return None
    answer = answer.strip()
    if not answer or "\n" in answer:
        return None
    # A newline terminates the answer even if later whitespace hit the token cap.
    if task == "gsm8k":
        return number(answer)
    if task == "arc":
        return answer if answer in (labels or []) else None
    if task == "boolq":
        return answer.lower() if answer.lower() in ("yes", "no") else None
    raise ValueError(task)


def score(text, row, truncated=False):
    answer = parse_answer(text, row["task"], row.get("labels"), truncated)
    expected = number(row["answer"]) if row["task"] == "gsm8k" else row["answer"]
    return int(answer is not None and answer == expected)


def user_prompt(row):
    instructions = {
        "gsm8k": "Solve the math problem. End with exactly one line 'Final answer: NUMBER'. Use only a number, decimal, or fraction after the marker.",
        "arc": "Choose the correct option. End with exactly one line 'Final answer: LABEL', using the original option label.",
        "boolq": "Answer the question using the passage. End with exactly one line 'Final answer: yes' or 'Final answer: no'.",
    }
    return instructions[row["task"]] + "\n\n" + row["question"]
