"""Model definitions."""

"""1-layer transformer for modular addition (TransformerLens)."""

from transformer_lens import HookedTransformer, HookedTransformerConfig


def build_model(p: int = 113, seed: int = 0, device="cpu"):
    """Build a 1-layer, no-LayerNorm, no-bias transformer for the mod-p addition task."""
    cfg = HookedTransformerConfig(
        n_layers=1, d_model=128, d_head=32, n_heads=4, d_mlp=512,
        d_vocab=p + 1,       # input tokens: 0..p-1 and "="
        d_vocab_out=p,       # outputs: only the p possible answers
        n_ctx=3,
        act_fn="relu",
        normalization_type=None,
        seed=seed,
        device=str(device),
    )
    model = HookedTransformer(cfg)

    # Freeze all biases at zero ("no biases").
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    return model, cfg