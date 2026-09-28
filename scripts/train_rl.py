"""Generate semi-random battle data, then behavior-clone it.

Run from the ai-cif repository root:

    uv run scripts/train_rl.py

The script deliberately has no CLI configuration. Important values are defined
in ``ai_cif.rl.config``, like the other training scripts in this repository.

There are two sequential phases:

    generate_battles()
    train_model()

Both are implemented in the ``ai_cif.rl`` package; this script is the entrypoint
that drives them.
"""

from ai_cif.rl import generate_battles, train_model


def main() -> None:
    generate_battles()
    train_model()


if __name__ == "__main__":
    main()
