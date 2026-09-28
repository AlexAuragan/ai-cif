"""Generate semi-random battle data, then behavior-clone it.

Battle generation uses only AsyncSemiRandomCombatHandler-derived policies.
The handler returns its ranking unchanged. It does NOT decide which fallback
action should count as the training label.

After a battle finishes, dataset generation resolves each recorded ranking
outside the handler: try action 1, then action 2, etc. and keep the first action
allowed by the tensorizer action mask. This mirrors the intended SDK fallback
semantics without making async_select_top_actions responsible for retry logic.

The pipeline is driven by scripts/train_rl.py, which runs the two sequential
phases:

    generate_battles()
    train_model()

Configuration lives in ``ai_cif.rl.config`` and is deliberately defined in code
rather than exposed through a CLI, like the other training scripts in this
repository.
"""

from ai_cif.rl.dataset import generate_battles
from ai_cif.rl.metrics import Metrics
from ai_cif.rl.recording import (
    RecordingSemiRandomHandler,
    encode_action,
    resolve_ranked_action,
    resolved_examples,
)
from ai_cif.rl.training import train_model

__all__ = [
    "Metrics",
    "RecordingSemiRandomHandler",
    "encode_action",
    "generate_battles",
    "resolve_ranked_action",
    "resolved_examples",
    "train_model",
]
