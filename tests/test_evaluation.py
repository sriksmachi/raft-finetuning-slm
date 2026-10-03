from lib.evaluation import (
    context_text,
    extract_answer,
    token_f1,
)


def test_answer_metrics_ignore_reasoning_wrapper() -> None:
    prediction = "Reasoning <ANSWER>: Patch monthly </ANSWER>"
    reference = "<ANSWER>Patch monthly</ANSWER>"

    assert extract_answer(prediction) == "Patch monthly"
    assert token_f1(prediction, reference) == 1.0


def test_context_text_flattens_nested_context() -> None:
    assert context_text({"sentences": [["Patch monthly."], ["Reboot after patching."]]}) == (
        "Patch monthly.\n\nReboot after patching."
    )
