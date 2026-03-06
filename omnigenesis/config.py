import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:
    raise RuntimeError(
        "PyYAML is required to load omnigenesis.yaml. Install with: pip install pyyaml"
    ) from exc


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_config_path() -> Path:
    return _repo_root() / "omnigenesis.yaml"


def _load_raw_config() -> tuple[Dict[str, Any], Path]:
    path = Path(os.getenv("OMNI_CONFIG_PATH", str(_default_config_path())))
    if not path.exists():
        return {}, path
    with path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}
    return _as_dict(parsed), path


def _detect_gpu_vram_gb() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return float(props.total_memory) / float(1024 ** 3)
    except Exception:
        return None


def _resolve_profile(raw: Dict[str, Any]) -> str:
    runtime = _as_dict(raw.get("runtime"))
    profiles = _as_dict(raw.get("profiles"))
    available = set(profiles.keys())

    requested = os.getenv("OMNI_PROFILE", runtime.get("active_profile", "auto"))
    if requested != "auto":
        if requested in available:
            return requested
        print(f"[WARN] Requested profile '{requested}' not found. Falling back to auto.", flush=True)

    auto = _as_dict(runtime.get("auto_profile"))
    cpu_profile = str(auto.get("cpu_profile", "small"))
    fallback_profile = str(auto.get("fallback_profile", cpu_profile))

    vram_gb = _detect_gpu_vram_gb()
    if vram_gb is None:
        return cpu_profile if cpu_profile in available else fallback_profile

    rules = auto.get("gpu_vram_gb_rules", [])
    if isinstance(rules, list):
        for rule in rules:
            rule_d = _as_dict(rule)
            max_vram = rule_d.get("max_vram_gb")
            profile = str(rule_d.get("profile", ""))
            try:
                max_vram_f = float(max_vram)
            except (TypeError, ValueError):
                continue
            if vram_gb <= max_vram_f and profile in available:
                return profile

    if fallback_profile in available:
        return fallback_profile
    if cpu_profile in available:
        return cpu_profile
    if "base" in available:
        return "base"
    return next(iter(available), "base")


def _build_effective_config(raw: Dict[str, Any], selected_profile: str) -> Dict[str, Any]:
    profiles = _as_dict(raw.get("profiles"))
    base_cfg = _as_dict(profiles.get("base"))
    profile_cfg = _as_dict(profiles.get(selected_profile))
    return _deep_merge(base_cfg, profile_cfg)


class AGIConfig:
    def __init__(self, values: Optional[Dict[str, Any]] = None):
        values = values or {}
        self.vocab_size = max(1, _as_int(values.get("vocab_size"), 50257))
        self.dim = max(32, _as_int(values.get("dim"), 192))
        self.heads = max(1, _as_int(values.get("heads"), 4))
        self.experts = max(1, _as_int(values.get("experts"), 4))
        self.max_reason_steps = max(0, _as_int(values.get("max_reason_steps"), 2))
        self.reason_threshold = _as_float(values.get("reason_threshold"), 0.85)
        self.novelty_threshold = _as_float(values.get("novelty_threshold"), 0.10)
        self.novelty_buf_size = max(1, _as_int(values.get("novelty_buf_size"), 512))
        self.novelty_sketch_dim = max(8, _as_int(values.get("novelty_sketch_dim"), 64))
        self.use_morton = _as_bool(values.get("use_morton"), True)
        self.morton_proj_dim = max(1, _as_int(values.get("morton_proj_dim"), 8))
        self.morton_bits = max(1, _as_int(values.get("morton_bits"), 6))

        if self.dim % self.heads != 0:
            raise ValueError(
                f"Invalid config: dim ({self.dim}) must be divisible by heads ({self.heads})."
            )
        if (self.dim // self.heads) % 2 != 0:
            raise ValueError(
                "Invalid config: head_dim must be even for RoPE "
                f"(dim={self.dim}, heads={self.heads})."
            )


class TrainConfig:
    def __init__(self, values: Optional[Dict[str, Any]] = None):
        values = values or {}
        self.seq_len = max(8, _as_int(values.get("seq_len"), 64))
        self.batch_size = max(1, _as_int(values.get("batch_size"), 1))
        self.grad_accum_steps = max(1, _as_int(values.get("grad_accum_steps"), 16))
        self.num_workers = max(0, _as_int(values.get("num_workers"), 0))
        self.prefetch_factor = max(2, _as_int(values.get("prefetch_factor"), 2))
        self.persistent_workers = _as_bool(values.get("persistent_workers"), False)
        self.pin_memory = _as_bool(values.get("pin_memory"), True)
        self.lr = _as_float(values.get("lr"), 2e-4)
        self.weight_decay = _as_float(values.get("weight_decay"), 0.01)
        self.grad_clip = _as_float(values.get("grad_clip"), 1.0)
        self.log_every = max(1, _as_int(values.get("log_every"), 25))
        self.ckpt_every = max(1, _as_int(values.get("ckpt_every"), 200))
        # 0 means unlimited.
        self.max_steps = _as_int(values.get("max_steps"), 0)


class DataConfig:
    def __init__(self, values: Optional[Dict[str, Any]] = None):
        values = values or {}
        self.dataset_name = str(values.get("dataset_name", "wikitext"))
        cfg_value = values.get("dataset_config", "wikitext-2-raw-v1")
        self.dataset_config = None if cfg_value in {"", None} else str(cfg_value)
        self.dataset_split = str(values.get("dataset_split", "train"))
        self.streaming = _as_bool(values.get("streaming"), False)
        self.text_field = str(values.get("text_field", "text"))
        # 0 means unlimited.
        self.max_examples = max(0, _as_int(values.get("max_examples"), 50000))
        # 0 means unlimited.
        self.max_chars_per_example = max(0, _as_int(values.get("max_chars_per_example"), 4000))


class InferenceConfig:
    def __init__(self, values: Optional[Dict[str, Any]] = None):
        values = values or {}
        self.max_new_tokens = max(1, _as_int(values.get("max_new_tokens"), 64))
        self.max_context_tokens = max(8, _as_int(values.get("max_context_tokens"), 64))
        self.do_sample = _as_bool(values.get("do_sample"), True)
        self.temperature = max(1e-5, _as_float(values.get("temperature"), 0.9))
        self.top_k = max(0, _as_int(values.get("top_k"), 50))
        top_p = _as_float(values.get("top_p"), 0.95)
        self.top_p = min(max(top_p, 0.0), 1.0)
        self.repetition_penalty = max(1.0, _as_float(values.get("repetition_penalty"), 1.1))


_raw_config, config_source_path = _load_raw_config()
active_profile = _resolve_profile(_raw_config)
_effective = _build_effective_config(_raw_config, active_profile)

cfg = AGIConfig(_as_dict(_effective.get("model")))
train_cfg = TrainConfig(_as_dict(_effective.get("train")))
data_cfg = DataConfig(_as_dict(_effective.get("data")))
inference_cfg = InferenceConfig(_as_dict(_effective.get("inference")))
