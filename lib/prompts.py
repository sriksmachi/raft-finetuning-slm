"""Shared prompt contract used by training and online inference."""

SYSTEM_PROMPT = (
    "You answer questions using only the supplied retrieved documents. "
    "Provide concise reasoning supported by quotes enclosed in ##begin_quote## and "
    "##end_quote##, then put the final answer inside <ANSWER></ANSWER>. "
    "If the documents do not contain enough evidence, state that the required "
    "information is missing."
)


def user_prompt(instruction: str) -> str:
    return f"<Retrieved Documents>:\n{instruction.strip()}"


def question_context_instruction(question: str, context: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Evaluation question must be non-empty text")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("Evaluation context must contain non-empty retrieved text")
    return f"<DOCUMENT>{context.strip()}</DOCUMENT>\n\n### Question:\n{question.strip()}"


def question_context_messages(question: str, context: str) -> list[dict[str, str]]:
    instruction = question_context_instruction(question, context)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(instruction)},
    ]
