import os
from typing import Iterable, Tuple

import pytest
import torch


def _to_list(keys: Iterable[str]) -> list[str]:
    return list(keys) if not isinstance(keys, list) else keys


# Mirror acceptance test choices but use full HF ids only (exclude TL-only configs)
PUBLIC_HF_MODELS = [
    "sshleifer/tiny-gpt2",
    "gpt2",
    "facebook/opt-125m",
    "EleutherAI/pythia-70m",
    "EleutherAI/gpt-neo-125M",
    "roneneldan/TinyStories-33M",
]

FULL_HF_MODELS = [
    "sshleifer/tiny-gpt2",
    "gpt2",
    "facebook/opt-125m",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",
    "bigcode/santacoder",
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "google/gemma-2b",
    "google/gemma-7b",
    "roneneldan/TinyStories-33M",
]

def _select_model_ids_from_acceptance_lists() -> list[str]:
    return FULL_HF_MODELS if os.environ.get("HF_TOKEN", "") else PUBLIC_HF_MODELS

# Allow overriding via env, comma-separated HF ids
DEFAULT_IDS = ",".join(_select_model_ids_from_acceptance_lists())
MODELS_ENV = os.getenv("TL_HOOK_SHAPE_MODELS", DEFAULT_IDS)
MODEL_NAMES = [m.strip() for m in MODELS_ENV.split(",") if m.strip()]


def _expected_shape_for_name(
    name: str,
    *,
    batch: int,
    pos: int,
    d_model: int,
    d_vocab: int | None,
    n_heads: int | None,
    d_head: int | None,
    d_mlp: int | None,
) -> Tuple[int, ...] | None:
    # Canonical TransformerBridge hook names only (no legacy aliases)

    # Embedding components
    if name.endswith("embed.hook_in") or name.endswith("pos_embed.hook_in"):
        return (batch, pos)
    if name.endswith("embed.hook_out") or name.endswith("pos_embed.hook_out"):
        return (batch, pos, d_model)

    # Unembedding
    if name.endswith("unembed.hook_in"):
        return (batch, pos, d_model)
    if name.endswith("unembed.hook_out") and d_vocab is not None:
        return (batch, pos, d_vocab)

    # Block IO
    if ".hook_in" in name and ".attn." not in name and ".mlp." not in name and ".ln" not in name:
        # blocks.{i}.hook_in
        return (batch, pos, d_model)
    if ".hook_out" in name and ".attn." not in name and ".mlp." not in name and ".ln" not in name:
        # blocks.{i}.hook_out
        return (batch, pos, d_model)

    # Attention module (canonical TB names)
    if name.endswith("attn.hook_in") or name.endswith("attn.hook_out"):
        return (batch, pos, d_model)
    if name.endswith("attn.hook_hidden_states"):
        return (batch, pos, d_model)
    if name.endswith("attn.hook_attention_weights") and n_heads is not None:
        return (batch, n_heads, pos, pos)

    # Attention subprojections: q/k/v/o
    if any(name.endswith(suf) for suf in ("attn.q.hook_in", "attn.k.hook_in", "attn.v.hook_in", "attn.o.hook_in")):
        return (batch, pos, d_model)
    if any(name.endswith(suf) for suf in ("attn.q.hook_out", "attn.k.hook_out", "attn.v.hook_out", "attn.o.hook_out")):
        return (batch, pos, d_model)

    # LayerNorms within blocks
    if ".ln" in name and name.endswith("hook_in"):
        return (batch, pos, d_model)
    if ".ln" in name and name.endswith("hook_out"):
        return (batch, pos, d_model)
    if name.endswith("hook_normalized"):
        return (batch, pos, d_model)
    if name.endswith("hook_scale"):
        # TB exposes full channel scale per position
        return (batch, pos, d_model)

    # MLP module
    if name.endswith("mlp.hook_in") or name.endswith("mlp.hook_out"):
        return (batch, pos, d_model)
    if name.endswith("mlp.hook_pre") and d_mlp is not None:
        return (batch, pos, d_mlp)

    return None


@pytest.mark.skipif(
    not os.getenv("TL_RUN_HOOK_SHAPE_TESTS"),
    reason="Set TL_RUN_HOOK_SHAPE_TESTS=1 to enable (requires internet + HF models)",
)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_transformer_bridge_hook_shapes(model_name: str):
    # Ensure boot method is registered
    from transformer_lens.model_bridge.bridge import TransformerBridge
    from transformer_lens.model_bridge.sources import (  # noqa: F401
        transformers as bridge_sources,
    )

    bridge = TransformerBridge.boot_transformers(model_name, device="cpu")

    prompt = "Hello world"
    tokens = bridge.to_tokens(prompt, move_to_device=False)
    batch, pos = int(tokens.shape[0]), int(tokens.shape[1])

    cfg = bridge.cfg
    d_model = int(getattr(cfg, "d_model"))
    d_vocab = int(getattr(cfg, "d_vocab", 0)) if hasattr(cfg, "d_vocab") else None
    n_heads = int(getattr(cfg, "n_heads", 0)) if hasattr(cfg, "n_heads") else None
    d_head = int(getattr(cfg, "d_head", 0)) if hasattr(cfg, "d_head") else None
    d_mlp = int(getattr(cfg, "d_mlp", 0)) if hasattr(cfg, "d_mlp") else None
    if n_heads == 0:
        n_heads = None
    if d_head == 0:
        d_head = None
    if d_mlp == 0:
        d_mlp = None

    _, cache = bridge.run_with_cache(tokens, device="cpu")
    keys = sorted(_to_list(cache.keys()))

    mismatches: list[tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    checked = 0
    for name in keys:
        exp = _expected_shape_for_name(
            name,
            batch=batch,
            pos=pos,
            d_model=d_model,
            d_vocab=d_vocab,
            n_heads=n_heads,
            d_head=d_head,
            d_mlp=d_mlp,
        )
        if exp is None:
            continue
        tensor = cache[name]
        assert isinstance(tensor, torch.Tensor), f"Non-tensor cached for {name}"
        got = tuple(tensor.shape)
        if got != exp:
            mismatches.append((name, exp, got))
        checked += 1

    assert checked > 0, "No hooks were checked; update expected mapping or model filter"
    msg = "\n".join(f"{n}: expected {e}, got {g}" for n, e, g in mismatches[:20])
    assert not mismatches, f"Found {len(mismatches)} shape mismatches. Examples:\n{msg}"


