"""Per-epoch behavior-cloning metrics."""

import torch


class Metrics:
    def __init__(self) -> None:
        self.counts = {
            name: [0, 0]
            for name in ("all", "choice", "move", "switch", "voluntary_switch")
        }

        self.ce_sum = 0.0
        self.choice_count = 0

    def add(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        force_switch: torch.Tensor,
        ce: torch.Tensor,
    ) -> None:
        choices = masks.sum(-1) > 1

        groups = {
            "all": torch.ones_like(choices),
            "choice": choices,
            "move": choices & (labels < 4),
            "switch": choices & (labels >= 4),
            "voluntary_switch": (choices & (labels >= 4) & ~force_switch),
        }

        correct = predictions == labels

        for name, selected in groups.items():
            self.counts[name][0] += int((correct & selected).sum().item())
            self.counts[name][1] += int(selected.sum().item())

        self.ce_sum += float(ce[choices].sum().item())

        self.choice_count += int(choices.sum().item())

    def result(self) -> dict:
        if self.choice_count == 0:
            raise ValueError(
                "No states with multiple legal actions "
                "were available for evaluation"
            )

        result = {"choice_ce": (self.ce_sum / self.choice_count)}

        for name, (correct, total) in self.counts.items():
            result[f"{name}_agreement"] = correct / total if total else None
            result[f"{name}_count"] = total

        return result
