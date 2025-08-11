# TransformerBridge Model Structure

This page describes the structure exposed by TransformerBridge, the canonical hook names to use, and the expected tensor shapes at each hook point.

## Overview

TransformerBridge wraps a Hugging Face model behind a consistent TransformerLens interface. It relies on:
- An ArchitectureAdapter that understands the HF module graph and provides a mapping to bridge components
- Generalized components (Embedding, Attention, MLP, Normalization, Block) exposing uniform hook points
- A light aliasing layer for backwards compatibility with legacy TransformerLens hook names

Construct a bridge from a HF model id:

```python
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources import transformers as bridge_sources  # registers boot

bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
```

You can then call the familiar APIs: `to_tokens`, `to_string`, `generate`, `run_with_hooks`, `run_with_cache`.

## Top-Level Components

Typical decoder-only models expose these top-level components (names vary by architecture):
- `embed`: token embedding
- `pos_embed` (if applicable) or rotary embeddings inside attention
- `blocks`: list-like container of transformer blocks
- `ln_final` (if applicable): final normalization
- `unembed`: output projection to vocabulary logits

Each `blocks.{i}` is a `BlockBridge` with subcomponents:
- `ln1`: normalization before attention
- `attn`: attention module
- `ln2`: normalization before MLP
- `mlp`: MLP module

## Canonical Hook Names

Use these canonical (non-aliased) names when adding hooks or reading from the cache.

### Embedding
- `embed.hook_in`: token ids (batch, pos)
- `embed.hook_out`: embeddings (batch, pos, d_model)
  - *Legacy alias: `hook_embed`*
- `pos_embed.hook_in` / `pos_embed.hook_out`: same shapes as above
  - *Legacy alias: `hook_pos_embed`*

### Residual stream
- `blocks.{i}.hook_in`: residual stream into block (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_resid_pre`*
- `blocks.{i}.hook_out`: residual stream out of block (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_resid_post`*
- `blocks.{i}.attn.hook_out`: residual stream after attention (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_resid_mid`*

### Attention
- `blocks.{i}.attn.hook_in`: (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_attn_in`*
- `blocks.{i}.attn.hook_out`: (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_attn_out`*
- `blocks.{i}.attn.hook_hidden_states`: primary output for caching (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.attn.hook_result`*
- `blocks.{i}.attn.hook_attention_weights` (and `hook_pattern`): (batch, n_heads, pos, pos)
  - *Legacy alias: `blocks.{i}.attn.hook_pattern`*
- When present, sub-projections: `blocks.{i}.attn.q/k/v/o.hook_in` / `.hook_out` (commonly (batch, pos, d_model))
  - *Legacy aliases: `blocks.{i}.hook_q_input`, `blocks.{i}.hook_k_input`, `blocks.{i}.hook_v_input`, `blocks.{i}.hook_q`, `blocks.{i}.hook_k`, `blocks.{i}.hook_v`*

### MLP
- `blocks.{i}.mlp.hook_in`: (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_mlp_in`*
- `blocks.{i}.mlp.hook_pre`: (batch, pos, d_mlp)
  - *Legacy alias: `blocks.{i}.hook_mlp_in` (via `mlp.in.hook_out`)*
- `blocks.{i}.mlp.hook_out`: (batch, pos, d_model)
  - *Legacy alias: `blocks.{i}.hook_mlp_out`*

### Normalization
- `blocks.{i}.ln1.hook_in` / `.hook_out`: (batch, pos, d_model)
  - *Legacy aliases for `.hook_out`: `blocks.{i}.ln1.hook_normalized`, `blocks.{i}.ln1.hook_scale`*
- Similarly for `ln2`
  - *Legacy aliases for `.hook_out`: `blocks.{i}.ln2.hook_normalized`, `blocks.{i}.ln2.hook_scale`*

### Unembedding / Logits
- `unembed.hook_in`: (batch, pos, d_model)
- `unembed.hook_out`: (batch, pos, d_vocab)

## Shapes at a Glance

- Residual stream and hidden states: (batch, pos, d_model)
- Attention patterns: (batch, n_heads, pos, pos)
- MLP pre-activation: (batch, pos, d_mlp)
- Embeddings: (batch, pos, d_model)
- Unembedding logits: (batch, pos, d_vocab)
- LayerNorm normalized / scale: (batch, pos, d_model)

These shapes are exercised in the multi-model shape test: `tests/integration/test_hook_shape_compatibility.py`.

## Booting from Hugging Face

`TransformerBridge.boot_transformers(model_id, ...)`:
- Loads the HF config/model/tokenizer
- Selects the appropriate ArchitectureAdapter
- Maps HF config fields to TransformerLens config (e.g., `d_model`, `n_heads`, `n_layers`, `d_mlp`, `d_vocab`, `n_ctx`, ...)
- Constructs the bridge and registers all hook points

## Fused QKV Attention

Some architectures use a fused QKV projection. The bridge's `JointQKVAttentionBridge` materializes `q`, `k`, and `v` `LinearBridge` views tied to the fused weights. Canonical attention hooks (`attn.hook_in/out`, `attn.hook_attention_weights`, etc.) retain the shapes listed above.

## Aliases and Backwards Compatibility

A minimal alias layer exists to ease migration from older TransformerLens names (e.g., `blocks.{i}.hook_resid_pre` → `blocks.{i}.hook_in`). New code should prefer the canonical names documented here.

## Example: Caching and Inspecting Hooks

```python
prompt = "Hello world"
logits, cache = bridge.run_with_cache(prompt)

# List some attention-related hooks on the first block
for k in cache.keys():
    if k.startswith("blocks.0.attn"):
        print(k, cache[k].shape)
```

For larger examples and a multi-model shape check, see `tests/integration/test_hook_shape_compatibility.py`.
