"""Training routines."""

"""Loss/accuracy helpers and the full-batch training loop."""

import os
import torch
from tqdm.auto import tqdm


def ce(final_logits, targets):
    """Cross-entropy on already-sliced [batch, p] logits (float64 for precision)."""
    log_probs = final_logits.double().log_softmax(dim=-1)
    return -log_probs.gather(1, targets[:, None]).mean()


def acc(final_logits, targets):
    """Accuracy on already-sliced [batch, p] logits."""
    return (final_logits.argmax(dim=-1) == targets).float().mean().item()


def loss_fn(logits, targets):
    """Cross-entropy using only the final '=' position of full [batch, seq, p] logits."""
    return ce(logits[:, -1, :], targets)


def accuracy_fn(logits, targets):
    """Accuracy using only the final '=' position of full [batch, seq, p] logits."""
    return acc(logits[:, -1, :], targets)


def train_model(model, optimizer, train_x, train_y, test_x, test_y,
                 num_epochs=12_000, log_every=100, ckpt_every=500,
                 checkpoint_dir="checkpoints"):
    """Full-batch training loop. Returns (history, saved_checkpoints)."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {k: [] for k in ["epoch", "train_loss", "test_loss", "train_acc", "test_acc", "weight_norm"]}
    saved_checkpoints = {}

    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        train_logits = model(train_x)
        train_loss = loss_fn(train_logits, train_y)
        train_loss.backward()
        optimizer.step()

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_x)
                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss.item())
                history["test_loss"].append(loss_fn(test_logits, test_y).item())
                history["train_acc"].append(accuracy_fn(train_logits, train_y))
                history["test_acc"].append(accuracy_fn(test_logits, test_y))
                history["weight_norm"].append(
                    sum(p.detach().double().pow(2).sum().item() for p in model.parameters()) ** 0.5
                )

        if epoch % ckpt_every == 0:
            saved_checkpoints[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(model.state_dict(), f"{checkpoint_dir}/grokked.pt")
    torch.save(saved_checkpoints, f"{checkpoint_dir}/all_checkpoints.pt")
    torch.save(history, "results/training_history.pt")

    return history, saved_checkpoints


def find_memorizing_epoch(history, saved_checkpoints, log_every=100,
                           train_acc_thresh=0.99, test_acc_thresh=0.1):
    """Earliest saved checkpoint where train acc is ~100% but test acc is still near chance."""
    idx = lambda e: e // log_every
    return next(
        e for e in sorted(saved_checkpoints)
        if history["train_acc"][idx(e)] >= train_acc_thresh
        and history["test_acc"][idx(e)] < test_acc_thresh
    )