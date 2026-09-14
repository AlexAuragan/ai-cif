from ai_cif.training.ppo import PPOMetrics


def print_metrics(metrics: PPOMetrics) -> None:
    print(
        f"policy={metrics.policy_loss:+.4f} "
        f"value={metrics.value_loss:.4f} "
        f"entropy={metrics.entropy:.4f} "
        f"kl={metrics.approx_kl:.5f} "
        f"clip={metrics.clip_fraction:.3f}"
    )

    print(
        f"mean_value={metrics.mean_value:+.3f} mean_return={metrics.mean_return:+.3f}"
    )


def print_initial_metrics(
    wins: int, losses: int, ties: int, eval_battles: int, evaluation_seconds: float
):
    initial_win_rate = wins / eval_battles

    print(f"wins={wins} losses={losses} ties={ties} win_rate={initial_win_rate:.1%}")

    print(
        f"evaluation_time="
        f"{evaluation_seconds:.2f}s "
        f"battles/s="
        f"{eval_battles / evaluation_seconds:.2f}"
    )
