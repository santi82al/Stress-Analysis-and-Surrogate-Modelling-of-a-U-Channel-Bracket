"""
Neural-network surrogate (PyTorch MLP) - alternative NN implementation.

Improvements over NN_sklearn_surrogate.py:
  - GROUP-AWARE early stopping: the stopping criterion is the loss on a
    validation set of ENTIRE held-out models (sklearn's early_stopping splits
    random rows, which leaks every model into the stopping signal).
  - SiLU activations + AdamW + cosine LR schedule, larger batches.
  - Trains on the full node set of the training models.

Same pooled formulation and canonical held-out-model test split (seed 0) as all
other scripts -> comparable numbers.

Run:  python NN_torch_surrogate.py
"""

import time

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

from surrogate_common import (load_dataset, heldout_split, evaluate_predictions,
                              report, FEATURE_NAMES)

HIDDEN = (256, 256, 128)
BATCH = 4096
MAX_EPOCHS = 120
PATIENCE = 15
LR = 1e-3
SEED = 0


def make_mlp(n_in, n_out):
    layers, prev = [], n_in
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


def train_net(Xtr, Ytr, Xval, Yval, verbose=True):
    """Standardize on train, fit with early stopping on the val loss.
    Returns (net, xscaler, yscaler, best_epoch)."""
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)

    Xt = torch.tensor(xs.transform(Xtr), dtype=torch.float32)
    Yt = torch.tensor(ys.transform(Ytr), dtype=torch.float32)
    Xv = torch.tensor(xs.transform(Xval), dtype=torch.float32, device=dev)
    Yv = torch.tensor(ys.transform(Yval), dtype=torch.float32, device=dev)

    ds = torch.utils.data.TensorDataset(Xt, Yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    net = make_mlp(Xtr.shape[1], Ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    lossf = nn.MSELoss()

    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        run = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item() * len(xb)
        sched.step()
        net.eval()
        with torch.no_grad():
            vloss = lossf(net(Xv), Yv).item()
        if verbose:
            print(f"epoch {epoch:3d}  train={run / len(ds):.5f}  val(models)={vloss:.5f}")
        if vloss < best_val - 1e-5:
            best_val, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop at epoch {epoch} (best {best_epoch})")
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, xs, ys, best_epoch


def predict(net, xs, ys, X):
    dev = next(net.parameters()).device
    with torch.no_grad():
        out = net(torch.tensor(xs.transform(X), dtype=torch.float32, device=dev))
    return ys.inverse_transform(out.cpu().numpy())


def split_val_models(groups_tr_idx, groups, frac=0.15, seed=1):
    """Carve validation MODELS out of the training index set (never the test set)."""
    rng = np.random.default_rng(seed)
    tr_models = np.unique(groups[groups_tr_idx])
    val_models = rng.choice(tr_models, size=max(2, int(frac * len(tr_models))),
                            replace=False)
    val_mask = np.isin(groups[groups_tr_idx], val_models)
    return groups_tr_idx[~val_mask], groups_tr_idx[val_mask]


if __name__ == "__main__":
    X, Y, groups, targets = load_dataset()
    tr, te = heldout_split(X, Y, groups)
    fit_idx, val_idx = split_val_models(tr, groups)
    print(f"train {len(np.unique(groups[fit_idx]))} models / "
          f"val {len(np.unique(groups[val_idx]))} models / "
          f"test {len(np.unique(groups[te]))} models")

    t0 = time.time()
    net, xs, ys, best_epoch = train_net(X[fit_idx], Y[fit_idx], X[val_idx], Y[val_idx])
    elapsed = time.time() - t0
    scores = evaluate_predictions(Y[te], predict(net, xs, ys, X[te]), targets)
    report("MLP_torch_256x256x128", scores, elapsed,
           notes=f"SiLU+AdamW+cosine, group-aware early stop (best epoch {best_epoch})")

    torch.save(net.state_dict(), "NN_torch_surrogate.pt")
    joblib.dump({"state_dict_file": "NN_torch_surrogate.pt", "hidden": HIDDEN,
                 "xscaler": xs, "yscaler": ys, "targets": targets,
                 "features": FEATURE_NAMES, "normalized": True},
                "NN_torch_surrogate_meta.joblib")
    print("saved -> NN_torch_surrogate.pt / NN_torch_surrogate_meta.joblib")
