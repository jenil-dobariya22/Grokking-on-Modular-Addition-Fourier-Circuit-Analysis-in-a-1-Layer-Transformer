"""Model analysis and visualization."""

"""Fourier analysis and causal ablations for the grokked circuit."""

import numpy as np
import torch


def fourier_norms(W):
    """W: [p, d] tensor/array. Returns [p] array: L2 norm over d of each DFT frequency."""
    W = W.numpy() if hasattr(W, "numpy") else W
    Fq = np.fft.fft(W, axis=0)
    return np.sqrt((np.abs(Fq) ** 2).sum(axis=1))


def folded_spectrum(norms, p):
    """Fold the DFT spectrum to frequencies 0..p//2 (real input has conjugate symmetry), drop DC."""
    spec = norms[: p // 2 + 1].copy()
    spec[0] = 0.0
    return spec


def key_frequencies(spec, p, n_std=2):
    """Frequencies whose spectrum value exceeds mean + n_std * std (DC excluded)."""
    thr = spec[1:].mean() + n_std * spec[1:].std()
    return [k for k in range(1, p // 2 + 1) if spec[k] > thr]


def fourier_filter_logits(final_logits, keep_freqs, p):
    """Keep only the given frequencies (+ conjugates) of logits over the answer axis. Drops DC."""
    Fq = torch.fft.fft(final_logits.double(), dim=1)
    mask = torch.zeros(p, dtype=torch.bool, device=final_logits.device)
    for f in keep_freqs:
        mask[f] = True
        mask[(-f) % p] = True
    return torch.fft.ifft(Fq * mask.to(Fq.dtype), dim=1).real


def project_embedding(W, freqs_to_keep, p, keep_dc=True):
    """Keep only selected Fourier frequencies (+ conjugates) in an embedding-like matrix."""
    W_np = W.numpy() if hasattr(W, "numpy") else W
    Fq = np.fft.fft(W_np, axis=0)
    mask = np.zeros(p, dtype=bool)
    if keep_dc:
        mask[0] = True
    for f in freqs_to_keep:
        mask[f] = True
        mask[(-f) % p] = True
    out = np.real(np.fft.ifft(Fq * mask[:, None], axis=0))
    return torch.tensor(out, dtype=torch.float32)


def eval_with_embedding(model, W_new, test_x, test_y, p, acc_fn):
    """Temporarily swap in a modified embedding, evaluate test accuracy, then restore it."""
    backup = model.W_E.data.clone()
    try:
        model.W_E.data[:p] = W_new.to(model.W_E.device)
        with torch.no_grad():
            logits = model(test_x)
        return acc_fn(logits[:, -1, :], test_y)
    finally:
        model.W_E.data.copy_(backup)


def key_energy_fraction(norms, key_freqs, p):
    """Fraction of non-DC Fourier energy contained in the given key frequencies."""
    e = norms ** 2
    e[0] = 0
    key_energy = sum(e[f] + e[p - f] for f in key_freqs)
    return key_energy / e.sum()