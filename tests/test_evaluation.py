from lib.evaluation import (
    context_text,
)


def test_context_text_flattens_nested_context() -> None:
    assert context_text({"sentences": [["Patch monthly."], ["Reboot after patching."]]}) == (
        "Patch monthly.\n\nReboot after patching."
    )
