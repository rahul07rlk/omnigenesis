import omnigenesis.data.streaming_dataset as streaming_dataset
from omnigenesis.data.streaming_dataset import ResumableStreamingDataset


class _DummyTokenizer:
    eos_token_id = 0

    def __call__(self, text, truncation=False, return_attention_mask=False, verbose=False):
        ids = [((ord(ch) % 97) + 1) for ch in text]
        return {"input_ids": ids}


def _patch_dataset(monkeypatch, rows):
    monkeypatch.setattr(streaming_dataset, "load_dataset", lambda *args, **kwargs: rows)


def _patch_dataset_error(monkeypatch, message):
    def _raise(*args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(streaming_dataset, "load_dataset", _raise)


def test_chat_message_extraction_and_format(monkeypatch):
    _patch_dataset(monkeypatch, [])
    ds = ResumableStreamingDataset(_DummyTokenizer(), seq_len=8)
    text = ds._extract_text(
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
    )
    assert "User: Hello" in text
    assert "Assistant: Hi there" in text


def test_english_filter_heuristic(monkeypatch):
    _patch_dataset(monkeypatch, [])
    ds = ResumableStreamingDataset(_DummyTokenizer(), seq_len=8, english_only=True)
    assert ds._is_english_like("This is an English sentence.")
    assert not ds._is_english_like("これは日本語の文です。")


def test_iter_yields_chunks_from_dialog_examples(monkeypatch):
    _patch_dataset(
        monkeypatch,
        [
            {"dialog": ["Hello there", "How are you today?", "I am doing fine."]},
        ],
    )
    ds = ResumableStreamingDataset(
        _DummyTokenizer(),
        seq_len=8,
        text_field="dialog",
        max_examples=1,
        max_chars_per_example=0,
    )
    x, y = next(iter(ds))
    assert x.shape == (8,)
    assert y.shape == (8,)


def test_script_dataset_failure_falls_back_to_builtin_corpus(monkeypatch):
    _patch_dataset_error(monkeypatch, "Dataset scripts are no longer supported, but found daily_dialog.py")
    ds = ResumableStreamingDataset(_DummyTokenizer(), seq_len=8, max_examples=1)
    x, y = next(iter(ds))
    assert x.shape == (8,)
    assert y.shape == (8,)


def test_builtin_english_chat_dataset_name(monkeypatch):
    _patch_dataset_error(monkeypatch, "should not be called")
    ds = ResumableStreamingDataset(
        _DummyTokenizer(),
        seq_len=8,
        max_examples=1,
        dataset_name="builtin_english_chat",
    )
    x, y = next(iter(ds))
    assert x.shape == (8,)
    assert y.shape == (8,)
