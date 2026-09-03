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
