"""
Shared PyTorch training/eval harness used by the project's deep models
(the TCN classifier used in this release), giving every model the same
discipline: same seeding convention, same optimizer/early-stopping
machinery, same train/val/test usage rule (early stopping and model
selection on validation loss only; the test set is touched exactly once,
after the best-epoch model is already fixed).

Hyperparameters here (documented, shared defaults for both models unless a
script overrides them explicitly, in which case that script documents the
override): Adam optimizer, lr=1e-3, weight_decay=1e-4, batch_size=128,
max_epochs=150, early_stopping_patience=15 (on validation BCE loss),
seed controls both torch's global RNG (weight init, dropout, batch
shuffling) and a dedicated numpy Generator.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_binary_classifier(model: nn.Module, X_train, y_train, X_val, y_val, X_test, y_test,
                             seed: int, max_epochs: int = 150, patience: int = 15,
                             lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 128,
                             verbose: bool = False):
    """X_* are already torch.float32 tensors of whatever shape the model
    expects (TCN: (n, ny, L); GNN: (n, N_NODES, L, 2)); y_* are float32
    tensors of 0/1 labels. Returns a result dict with test predictions,
    probabilities, training history, best epoch, training time, and
    parameter count. The TEST tensors are never touched until after the
    early-stopping loop has already selected and restored the best-val-loss
    model state -- verified structurally here (test_* is not referenced
    anywhere inside the training loop below) and re-checked by
    tests/test_deep_models.py."""
    set_all_seeds(seed)
    device = torch.device("cpu")
    model = model.to(device)

    train_ds = TensorDataset(X_train, y_train)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()
        history.append(dict(epoch=epoch, val_loss=val_loss))

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epochs_since_improve >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}, best_epoch={best_epoch}, "
                      f"best_val_loss={best_val_loss:.4f}")
            break
    training_time_s = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_probs = torch.sigmoid(test_logits).numpy()
    test_preds = (test_probs >= 0.5).astype(int)

    return dict(
        test_probs=test_probs.tolist(),
        test_preds=test_preds.tolist(),
        test_labels=y_test.numpy().astype(int).tolist(),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        n_epochs_run=len(history),
        history=history,
        training_time_s=training_time_s,
        param_count=model.param_count() if hasattr(model, "param_count") else
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        seed=seed,
    )
