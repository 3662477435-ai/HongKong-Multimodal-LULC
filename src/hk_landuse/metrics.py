from __future__ import annotations

import numpy as np
import torch

from .labels import CLASS_NAMES, IGNORE_INDEX


class ConfusionMetrics:
    """Streaming confusion matrix with complete per-class diagnostics."""

    def __init__(
        self,
        num_classes: int = 10,
        class_names: list[str] | tuple[str, ...] | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = list(class_names or CLASS_NAMES)
        if len(self.class_names) != num_classes:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        pred = prediction.argmax(dim=1) if prediction.ndim == target.ndim + 1 else prediction
        pred = pred.detach().cpu().long()
        target = target.detach().cpu().long()
        valid = target != self.ignore_index
        valid &= (target >= 0) & (target < self.num_classes)
        if mask is not None:
            valid &= mask.detach().cpu().bool()
        flat = target[valid] * self.num_classes + pred[valid]
        if flat.numel():
            self.matrix += torch.bincount(flat, minlength=self.num_classes**2).reshape(
                self.num_classes, self.num_classes
            )

    def compute(self) -> dict:
        matrix = self.matrix.numpy()
        tp = np.diag(matrix).astype(float)
        support = matrix.sum(axis=1).astype(float)
        predicted = matrix.sum(axis=0).astype(float)
        union = support + predicted - tp
        precision = np.divide(tp, predicted, out=np.full_like(tp, np.nan), where=predicted > 0)
        recall = np.divide(tp, support, out=np.full_like(tp, np.nan), where=support > 0)
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.full_like(tp, np.nan),
            where=(precision + recall) > 0,
        )
        row_normalized = np.divide(
            matrix,
            support[:, None],
            out=np.zeros_like(matrix, dtype=float),
            where=support[:, None] > 0,
        )
        present = support > 0
        return {
            "overall_accuracy": float(tp.sum() / max(matrix.sum(), 1)),
            "macro_iou_present": float(np.nanmean(iou[present])) if present.any() else float("nan"),
            "macro_f1_present": float(np.nanmean(f1[present])) if present.any() else float("nan"),
            "macro_precision_present": float(np.nanmean(precision[present])) if present.any() else float("nan"),
            "macro_recall_present": float(np.nanmean(recall[present])) if present.any() else float("nan"),
            "per_class": {
                self.class_names[i]: {
                    "support": int(support[i]),
                    "predicted": int(predicted[i]),
                    "precision": None if np.isnan(precision[i]) else float(precision[i]),
                    "recall": None if np.isnan(recall[i]) else float(recall[i]),
                    "f1": None if np.isnan(f1[i]) else float(f1[i]),
                    "iou": None if np.isnan(iou[i]) else float(iou[i]),
                }
                for i in range(self.num_classes)
            },
            "confusion_matrix": matrix.tolist(),
            "confusion_matrix_row_normalized": row_normalized.tolist(),
        }
