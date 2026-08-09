from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .metrics import classification_metrics


def nll_from_probs(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.nll_loss(probs.clamp_min(1e-8).log(), target)


def _move(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["backbone"].to(device, non_blocking=True),
        batch["clip"].to(device, non_blocking=True),
        batch["target"].to(device, non_blocking=True),
    )


def train_one_epoch(model, loader, optimizer, device, scaler=None) -> float:
    model.train()
    running = 0.0
    seen = 0
    use_amp = scaler is not None and device.type == "cuda"
    for batch in tqdm(loader, desc="train", leave=False):
        backbone_x, clip_x, target = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(backbone_x, clip_x)
            loss = nll_from_probs(out["probs"], target)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = target.size(0)
        running += float(loss.detach()) * bs
        seen += bs
    return running / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    ys, preds, probs = [], [], []
    total_loss = 0.0
    seen = 0
    start = time.perf_counter()
    for batch in tqdm(loader, desc="eval", leave=False):
        backbone_x, clip_x, target = _move(batch, device)
        out = model(backbone_x, clip_x)
        p = out["probs"]
        loss = nll_from_probs(p, target)
        bs = target.size(0)
        total_loss += float(loss) * bs
        seen += bs
        ys.extend(target.cpu().tolist())
        preds.extend(p.argmax(dim=1).cpu().tolist())
        probs.append(p.cpu().numpy())
    elapsed = time.perf_counter() - start
    probs_np = np.concatenate(probs, axis=0)
    metrics = classification_metrics(ys, preds, probs_np)
    metrics["loss"] = total_loss / max(seen, 1)
    metrics["elapsed_seconds"] = elapsed
    metrics["seconds_per_sample"] = elapsed / max(seen, 1)
    return metrics, {"y_true": np.asarray(ys), "y_pred": np.asarray(preds), "y_prob": probs_np}


def fit(model, data, cfg, device, output_dir: str | Path) -> dict:
    tcfg = cfg["training"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg.get("lr", 2e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    epochs = int(tcfg.get("epochs", 60))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = device.type == "cuda" and bool(tcfg.get("amp", True))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # compatibility with older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_acc = -1.0
    best_epoch = -1
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, data.train, optimizer, device, scaler)
        val_metrics, _ = evaluate(model, data.val, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f}")
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_names": data.class_names,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )
    return {"best_val_accuracy": best_acc, "best_epoch": best_epoch, "history": history}
