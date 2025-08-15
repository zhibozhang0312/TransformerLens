from transformer_lens.model_bridge import TransformerBridge

MODEL = "gpt2"

prompt = "Hello World!"
bridge = TransformerBridge.boot_transformers(
    MODEL,
    device="cpu",
    hf_config_overrides={
        "attn_implementation": "eager",
    },
)

act_names_in_cache = [
    # "hook_embed",
    # "hook_pos_embed",
    "blocks.0.hook_resid_pre",
    "blocks.0.ln1.hook_scale",
    "blocks.0.ln1.hook_normalized",
    "blocks.0.attn.hook_q",
    "blocks.0.attn.hook_k",
    "blocks.0.attn.hook_v",
    "blocks.0.attn.hook_attn_scores",
    "blocks.0.attn.hook_pattern",
    "blocks.0.attn.hook_z",
    "blocks.0.hook_attn_out",
    "blocks.0.hook_resid_mid",
    "blocks.0.ln2.hook_scale",
    "blocks.0.ln2.hook_normalized",
    "blocks.0.mlp.hook_pre",
    # "blocks.0.mlp.hook_post",
    "blocks.0.hook_mlp_out",
    "blocks.0.hook_resid_post",
    "ln_final.hook_scale",
    "ln_final.hook_normalized",
]


def test_cache_hook_names():
    """Test that TransformerBridge cache contains the expected hook names."""
    _, cache = bridge.run_with_cache(prompt)

    # Get the actual cache keys
    actual_keys = list(cache.keys())

    print(f"\nExpected hooks: {len(act_names_in_cache)}")
    print(f"Actual hooks: {len(actual_keys)}")

    # Find missing and extra hooks
    expected_set = set(act_names_in_cache)
    actual_set = set(actual_keys)

    missing_hooks = expected_set - actual_set
    extra_hooks = actual_set - expected_set

    print(f"Missing hooks ({len(missing_hooks)}): {sorted(missing_hooks)}")
    print(
        f"Extra hooks ({len(extra_hooks)}): {sorted(list(extra_hooks)[:10])}{'...' if len(extra_hooks) > 10 else ''}"
    )

    # Check that all expected hooks are present (subset check)
    # It's okay to have extra hooks - that means more functionality is exposed
    assert len(missing_hooks) == 0, f"Missing expected hooks: {sorted(missing_hooks)}"

    # Verify we have at least the expected hooks
    assert all(
        hook in actual_set for hook in expected_set
    ), f"Some expected hooks are missing: {missing_hooks}"
