from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset


class ResumableStreamingDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_len: int = 64,
        skip_seqs: int = 0,
        resume_state: Optional[dict] = None,
        dataset_name: str = "wikitext",
        dataset_config: Optional[str] = "wikitext-2-raw-v1",
        dataset_split: str = "train",
        streaming: bool = False,
        text_field: str = "text",
        max_examples: int = 0,
        max_chars_per_example: int = 0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self.max_examples = max_examples if max_examples > 0 else None
        self.max_chars_per_example = max_chars_per_example if max_chars_per_example > 0 else None

        resume_state = resume_state or {}
        self._buffer = list(resume_state.get("buffer", []))
        self._seqs_emitted = int(resume_state.get("seqs_emitted", 0))
        self._examples_seen = int(resume_state.get("examples_seen", 0))
        self._skip_remaining = int(resume_state.get("skip_remaining", skip_seqs))

        ds_kwargs = {"split": dataset_split, "streaming": streaming}
        if dataset_config is not None and dataset_config != "":
            ds_kwargs["name"] = dataset_config
        self.dataset = load_dataset(dataset_name, **ds_kwargs)
        self._example_skip_fallback = 0
        if self._examples_seen > 0:
            skip_fn = getattr(self.dataset, "skip", None)
            if callable(skip_fn):
                self.dataset = skip_fn(self._examples_seen)
            else:
                self._example_skip_fallback = self._examples_seen

    def _snapshot_state(
        self,
        buffer: list,
        seqs_emitted: int,
        examples_seen: int,
        skip_remaining: int,
    ) -> None:
        self._buffer = list(buffer)
        self._seqs_emitted = int(seqs_emitted)
        self._examples_seen = int(examples_seen)
        self._skip_remaining = int(skip_remaining)

    def get_resume_state(self) -> dict:
        return {
            "buffer": list(self._buffer),
            "seqs_emitted": int(self._seqs_emitted),
            "examples_seen": int(self._examples_seen),
            "skip_remaining": int(self._skip_remaining),
        }

    def __iter__(self):
        buffer = list(self._buffer)
        seqs_emitted = int(self._seqs_emitted)
        examples_seen = int(self._examples_seen)
        skip_remaining = int(self._skip_remaining)
        eos_token_id = self.tokenizer.eos_token_id

        for example in self.dataset:
            if self.max_examples is not None and examples_seen >= self.max_examples:
                self._snapshot_state(buffer, seqs_emitted, examples_seen, skip_remaining)
                break

            if self._example_skip_fallback > 0:
                self._example_skip_fallback -= 1
                continue

            text = example.get(self.text_field, "") if isinstance(example, dict) else ""
            if not isinstance(text, str):
                text = str(text)
            if self.max_chars_per_example is not None and len(text) > self.max_chars_per_example:
                text = text[: self.max_chars_per_example]

            tokens = self.tokenizer(
                text,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            if eos_token_id is not None:
                tokens.append(eos_token_id)
            buffer.extend(tokens)
            examples_seen += 1

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                seqs_emitted += 1

                if skip_remaining > 0:
                    skip_remaining -= 1
                    self._snapshot_state(buffer, seqs_emitted, examples_seen, skip_remaining)
                    continue

                self._snapshot_state(buffer, seqs_emitted, examples_seen, skip_remaining)
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )
            self._snapshot_state(buffer, seqs_emitted, examples_seen, skip_remaining)
