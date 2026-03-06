from typing import Iterable, Optional

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info


class _BuiltinEnglishCorpus:
    def __init__(
        self,
        rows: list[dict],
        offset: int = 0,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        self._rows = rows
        self._offset = max(0, int(offset))
        self._num_shards = max(1, int(num_shards))
        self._shard_index = max(0, min(int(shard_index), self._num_shards - 1))

    def __iter__(self):
        idx = self._offset
        n = len(self._rows)
        while True:
            if (idx % self._num_shards) == self._shard_index:
                yield self._rows[idx % n]
            idx += 1

    def skip(self, n: int):
        return _BuiltinEnglishCorpus(
            self._rows,
            offset=self._offset + max(0, int(n)),
            num_shards=self._num_shards,
            shard_index=self._shard_index,
        )

    def shard(self, num_shards: int, index: int):
        return _BuiltinEnglishCorpus(
            self._rows,
            offset=self._offset,
            num_shards=max(1, int(num_shards)),
            shard_index=max(0, int(index)),
        )


class ResumableStreamingDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_len: int = 64,
        skip_seqs: int = 0,
        resume_state: Optional[dict] = None,
        dataset_name: str = "daily_dialog",
        dataset_config: Optional[str] = None,
        dataset_split: str = "train",
        streaming: bool = False,
        text_field: str = "dialog",
        max_examples: int = 0,
        max_chars_per_example: int = 0,
        english_only: bool = True,
        min_english_ratio: float = 0.70,
        chat_messages_field: str = "messages",
        chat_role_field: str = "role",
        chat_content_field: str = "content",
        chat_role_fallback_fields: Optional[list[str]] = None,
        chat_content_fallback_fields: Optional[list[str]] = None,
        prompt_field: str = "prompt",
        response_field: str = "response",
        prompt_fallback_fields: Optional[list[str]] = None,
        response_fallback_fields: Optional[list[str]] = None,
        max_turns_per_example: int = 0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self.max_examples = max_examples if max_examples > 0 else None
        self.max_chars_per_example = max_chars_per_example if max_chars_per_example > 0 else None
        self.english_only = bool(english_only)
        self.min_english_ratio = min(max(float(min_english_ratio), 0.0), 1.0)
        self.chat_messages_field = str(chat_messages_field)
        self.chat_role_field = str(chat_role_field)
        self.chat_content_field = str(chat_content_field)
        self.chat_role_fallback_fields = tuple(chat_role_fallback_fields or ["from", "speaker"])
        self.chat_content_fallback_fields = tuple(
            chat_content_fallback_fields or ["value", "text", "utterance"]
        )
        self.prompt_field = str(prompt_field)
        self.response_field = str(response_field)
        self.prompt_fallback_fields = tuple(
            prompt_fallback_fields or ["instruction", "question", "query", "input"]
        )
        self.response_fallback_fields = tuple(
            response_fallback_fields or ["response", "output", "answer", "completion"]
        )
        self.max_turns_per_example = max(0, int(max_turns_per_example))

        resume_state = resume_state or {}
        self._buffer = list(resume_state.get("buffer", []))
        self._seqs_emitted = int(resume_state.get("seqs_emitted", 0))
        self._examples_seen = int(resume_state.get("examples_seen", 0))
        self._skip_remaining = int(resume_state.get("skip_remaining", skip_seqs))

        ds_kwargs = {"split": dataset_split, "streaming": streaming}
        if dataset_config is not None and dataset_config != "":
            ds_kwargs["name"] = dataset_config
        self.dataset = self._load_dataset_or_fallback(dataset_name, ds_kwargs)
        self._example_skip_fallback = 0
        if self._examples_seen > 0:
            skip_fn = getattr(self.dataset, "skip", None)
            if callable(skip_fn):
                self.dataset = skip_fn(self._examples_seen)
            else:
                self._example_skip_fallback = self._examples_seen

    def _builtin_rows(self) -> list[dict]:
        return [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help you today?"}]},
            {"messages": [{"role": "user", "content": "Explain photosynthesis simply."}, {"role": "assistant", "content": "Plants use sunlight, water, and carbon dioxide to make food and release oxygen."}]},
            {"messages": [{"role": "user", "content": "What is RAM versus storage?"}, {"role": "assistant", "content": "RAM is short-term working memory, while storage keeps files long-term."}]},
            {"messages": [{"role": "user", "content": "Give three healthy breakfast ideas."}, {"role": "assistant", "content": "Oatmeal with fruit, eggs with whole-grain toast, and yogurt with nuts."}]},
            {"messages": [{"role": "user", "content": "How can I debug Python code?"}, {"role": "assistant", "content": "Reproduce the bug, inspect stack traces, add logs, and test small units."}]},
            {"messages": [{"role": "user", "content": "What is overfitting?"}, {"role": "assistant", "content": "Overfitting means memorizing training data and failing on new data."}]},
            {"messages": [{"role": "user", "content": "Write a polite extension request email."}, {"role": "assistant", "content": "Hello, I am requesting a short extension due to an unexpected delay. Thank you for your consideration."}]},
            {"messages": [{"role": "user", "content": "Summarize: Exercise improves cardiovascular health."}, {"role": "assistant", "content": "Exercise helps your heart and blood vessels work better."}]},
            {"messages": [{"role": "user", "content": "What is a REST API?"}, {"role": "assistant", "content": "A REST API lets software communicate using HTTP methods and structured responses."}]},
            {"messages": [{"role": "user", "content": "How do I stay safe online?"}, {"role": "assistant", "content": "Use strong passwords, enable two-factor auth, and avoid suspicious links."}]},
        ]

    def _load_dataset_or_fallback(self, dataset_name: str, ds_kwargs: dict):
        normalized = str(dataset_name).strip().lower()
        if normalized in {"builtin_english_chat", "internal_english_chat"}:
            print("[INFO] Using built-in English chat corpus.", flush=True)
            return _BuiltinEnglishCorpus(self._builtin_rows())
        try:
            return load_dataset(dataset_name, **ds_kwargs)
        except Exception as exc:
            print(
                f"[WARN] Failed to load dataset {dataset_name} with error: {exc}",
                flush=True,
            )
            if "Dataset scripts are no longer supported" in str(exc):
                print(
                    "[WARN] This datasets version blocks script-based datasets. "
                    "Falling back to internal English chat corpus.",
                    flush=True,
                )
            else:
                print("[WARN] Falling back to internal English chat corpus.", flush=True)
            return _BuiltinEnglishCorpus(self._builtin_rows())

    def _stringify(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                text = self._stringify(item)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        if isinstance(value, dict):
            if value.get("type") == "text" and "text" in value:
                return self._stringify(value.get("text"))
            for key in ("content", "text", "value", "output"):
                if key in value:
                    text = self._stringify(value.get(key))
                    if text:
                        return text
            return str(value)
        return str(value)

    def _value_from_keys(self, example: dict, keys: Iterable[str]):
        for key in keys:
            if key in example:
                return example.get(key)
        return None

    def _role_label(self, value) -> str:
        role = str(value or "").strip().lower()
        if role in {"assistant", "bot", "model", "gpt", "chatgpt"}:
            return "Assistant"
        if role in {"user", "human", "prompter", "customer", "client"}:
            return "User"
        if role == "system":
            return "System"
        if role:
            return role.capitalize()
        return "Message"

    def _extract_messages_text(self, example: dict) -> str:
        messages = self._value_from_keys(
            example,
            (
                self.chat_messages_field,
                "messages",
                "conversations",
                "conversation",
                "dialog",
                "dialogue",
                "chat",
            ),
        )
        if not isinstance(messages, list):
            return ""

        lines: list[str] = []
        for idx, message in enumerate(messages):
            if isinstance(message, str):
                text = message.strip()
                if text:
                    speaker = "User" if (idx % 2 == 0) else "Assistant"
                    lines.append(f"{speaker}: {text}")
                continue
            if not isinstance(message, dict):
                continue
            role = self._value_from_keys(message, (self.chat_role_field, *self.chat_role_fallback_fields))
            content = self._value_from_keys(
                message,
                (self.chat_content_field, *self.chat_content_fallback_fields),
            )
            text = self._stringify(content).strip()
            if text:
                lines.append(f"{self._role_label(role)}: {text}")

        if self.max_turns_per_example > 0 and len(lines) > self.max_turns_per_example:
            lines = lines[-self.max_turns_per_example :]
        return "\n".join(lines)

    def _extract_pair_text(self, example: dict) -> str:
        prompt = self._value_from_keys(example, (self.prompt_field, *self.prompt_fallback_fields))
        response = self._value_from_keys(example, (self.response_field, *self.response_fallback_fields))
        p_text = self._stringify(prompt).strip()
        r_text = self._stringify(response).strip()
        if p_text and r_text:
            return f"User: {p_text}\nAssistant: {r_text}"
        if r_text:
            return f"Assistant: {r_text}"
        return ""

    def _extract_text(self, example) -> str:
        if isinstance(example, str):
            return example
        if not isinstance(example, dict):
            return self._stringify(example)

        chat_text = self._extract_messages_text(example)
        if chat_text:
            return chat_text

        pair_text = self._extract_pair_text(example)
        if pair_text:
            return pair_text

        direct = self._value_from_keys(example, (self.text_field, "text", "content", "document"))
        return self._stringify(direct)

    def _is_english_like(self, text: str) -> bool:
        if not self.english_only:
            return True
        letters = 0
        latin_letters = 0
        for ch in text:
            if ch.isalpha():
                letters += 1
                if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                    latin_letters += 1
        if letters == 0:
            return False
        return (latin_letters / letters) >= self.min_english_ratio

    def _worker_dataset(self):
        worker = get_worker_info()
        if worker is None or worker.num_workers <= 1:
            return self.dataset, self._example_skip_fallback, None
        shard_fn = getattr(self.dataset, "shard", None)
        if callable(shard_fn):
            try:
                worker_ds = shard_fn(num_shards=worker.num_workers, index=worker.id)
                return worker_ds, self._example_skip_fallback, None
            except Exception:
                pass
        return self.dataset, self._example_skip_fallback, worker

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
        dataset, example_skip_fallback, worker = self._worker_dataset()
        buffer = list(self._buffer)
        seqs_emitted = int(self._seqs_emitted)
        examples_seen = int(self._examples_seen)
        skip_remaining = int(self._skip_remaining)
        eos_token_id = self.tokenizer.eos_token_id

        for row_idx, example in enumerate(dataset):
            if self.max_examples is not None and examples_seen >= self.max_examples:
                self._snapshot_state(buffer, seqs_emitted, examples_seen, skip_remaining)
                break

            if worker is not None and (row_idx % worker.num_workers) != worker.id:
                continue

            if example_skip_fallback > 0:
                example_skip_fallback -= 1
                continue

            text = self._extract_text(example).strip()
            if not text:
                continue
            if not self._is_english_like(text):
                continue
            if self.max_chars_per_example is not None and len(text) > self.max_chars_per_example:
                text = text[: self.max_chars_per_example]

            tokens = self.tokenizer(
                text,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
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
