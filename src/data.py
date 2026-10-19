"""Dataset generation and preparation."""

"""Dataset construction for (a + b) mod p."""

import torch


def build_dataset(p: int = 113, frac_train: float = 0.3, seed: int = 0, device="cpu"):
    """Build the full (a, b) -> (a+b) mod p dataset and a seeded train/test split.

    Returns a dict with: train_x, train_y, test_x, test_y, all_x, all_y, pairs, labels.
    """
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))  # [p*p, 2]
    labels = (pairs[:, 0] + pairs[:, 1]) % p

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=g)
    split = int(frac_train * len(pairs))
    train_idx, test_idx = perm[:split], perm[split:]

    def make_x(idx):
        ab = pairs[idx]
        eq = torch.full((len(idx), 1), p)  # token id p = "="
        return torch.cat([ab, eq], dim=1).to(device)

    train_x, train_y = make_x(train_idx), labels[train_idx].to(device)
    test_x, test_y = make_x(test_idx), labels[test_idx].to(device)
    all_x = make_x(torch.arange(len(pairs)))
    all_y = labels.to(device)

    return {
        "train_x": train_x, "train_y": train_y,
        "test_x": test_x, "test_y": test_y,
        "all_x": all_x, "all_y": all_y,
        "pairs": pairs, "labels": labels,
    }