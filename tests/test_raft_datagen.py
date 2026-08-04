import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


def _import_raft_datagen(monkeypatch):
    azure_identity = SimpleNamespace(
        AzureCliCredential=MagicMock(),
        get_bearer_token_provider=MagicMock(return_value=lambda: "token"),
    )
    monkeypatch.setitem(sys.modules, "azure.identity", azure_identity)
    sys.modules.pop("lib.raft_datagen", None)
    from lib import raft_datagen

    return raft_datagen


def test_validate_special_characters(monkeypatch) -> None:
    raft_datagen = _import_raft_datagen(monkeypatch)
    special_characters = lambda s: raft_datagen.remove_special_characters(s)

    assert special_characters("Hello, World..................") == "Hello World"
    assert special_characters("This is a test.") == "This is a test"
    assert special_characters("Special characters: @#$%^&*()") == "Special characters"
    assert special_characters("No special characters") == "No special characters"
    assert special_characters("  \t\n\u2003  ") == ""


def test_load_chunks_excludes_whitespace_only_cleaned_text(monkeypatch, tmp_path) -> None:
    raft_datagen = _import_raft_datagen(monkeypatch)
    (tmp_path / "symbols.txt").write_text("@ " * 300, encoding="utf-8")
    (tmp_path / "valid.txt").write_text("valid   content " * 20, encoding="utf-8")

    chunks = raft_datagen.load_chunks_from_dir(tmp_path)

    assert chunks == ["valid content " * 19 + "valid content"]


def test_generate_dataset_processes_chunks_sequentially(monkeypatch) -> None:
    raft_datagen = _import_raft_datagen(monkeypatch)
    processed_chunks = []

    def generate_questions(chunk, _num_questions):
        processed_chunks.append(chunk)
        return [f"question for {chunk}"]

    monkeypatch.setattr(raft_datagen, "generate_instructions_gen", generate_questions)
    monkeypatch.setattr(
        raft_datagen,
        "generate_label",
        lambda question, _chunk: f"<ANSWER>{question}</ANSWER>",
    )
    monkeypatch.setattr(
        raft_datagen,
        "build_raft_context_docs",
        lambda chunks, index, _num_distract, _p: ([chunks[index]], True),
    )

    raft_datagen.generate_dataset(
        ["first chunk", "second chunk"],
        num_questions=1,
        num_distract=0,
    )

    assert processed_chunks == ["first chunk", "second chunk"]
    assert raft_datagen.ds["oracle_context"] == ["first chunk", "second chunk"]