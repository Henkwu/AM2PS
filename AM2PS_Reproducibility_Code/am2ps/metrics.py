from __future__ import annotations

from typing import Sequence
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = y_prob.shape[1]
    average = "binary" if n_classes == 2 else "macro"
    kwargs = {"average": average, "zero_division": 0}
    if average == "binary":
        kwargs["pos_label"] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, **kwargs)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    try:
        if n_classes == 2:
            out["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            out["auc_ovr_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        pass
    return out
