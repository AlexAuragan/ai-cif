from math import log

import numpy as np
from sklearn.tree import DecisionTreeRegressor

ACTION_COUNT = 10
PROBABILITY_EPSILON = 1e-8


class DecisionTreePolicy:
    def __init__(
        self,
        *,
        action_count: int = ACTION_COUNT,
        max_depth: int | None = None,
        max_leaf_nodes: int = 32,
        seed: int = 42,
    ) -> None:
        if action_count <= 1:
            raise ValueError("action_count must be > 1")

        self.action_count = action_count
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.seed = seed

        self.tree: DecisionTreeRegressor | None = None
        self._fit_count = 0
        self._rng = np.random.default_rng(seed)

    def raw_probabilities_batch(
        self, features: np.ndarray, *, tree: DecisionTreeRegressor | None = None
    ) -> np.ndarray:
        selected_tree = self.tree if tree is None else tree

        probabilities = self._raw_probabilities(features, tree=selected_tree)

        totals = probabilities.sum(axis=1, keepdims=True)

        if np.any(totals <= 0.0):
            raise RuntimeError("Tree returned invalid probabilities")

        return probabilities / totals

    def _raw_probabilities(
        self, features: np.ndarray, *, tree: DecisionTreeRegressor | None
    ) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, feature_count]")

        if tree is None:
            return np.full(
                (features.shape[0], self.action_count),
                1.0 / self.action_count,
                dtype=np.float64,
            )

        prediction = np.asarray(tree.predict(features), dtype=np.float64)

        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)

        expected_shape = (features.shape[0], self.action_count)

        if prediction.shape != expected_shape:
            raise RuntimeError(
                f"Tree returned shape {prediction.shape}, expected {expected_shape}"
            )

        return np.clip(prediction, PROBABILITY_EPSILON, None)

    def probabilities_batch(
        self,
        features: np.ndarray,
        action_mask: np.ndarray,
        *,
        tree: DecisionTreeRegressor | None = None,
    ) -> np.ndarray:
        if action_mask.shape != (features.shape[0], self.action_count):
            raise ValueError(
                "action_mask must have shape [batch, action_count]"
            )

        probabilities = self.raw_probabilities_batch(features, tree=tree)

        probabilities = np.where(action_mask, probabilities, 0.0)

        totals = probabilities.sum(axis=1, keepdims=True)

        if np.any(totals <= 0.0):
            raise RuntimeError("Policy received a state with no legal actions")

        return probabilities / totals

    def probabilities(
        self, features: np.ndarray, action_mask: np.ndarray
    ) -> np.ndarray:
        if features.ndim != 1:
            raise ValueError("features must be 1-D")
        if action_mask.shape != (self.action_count,):
            raise ValueError("action_mask has the wrong shape")

        return self.probabilities_batch(
            features.reshape(1, -1), action_mask.reshape(1, -1)
        )[0]

    def sample(
        self, features: np.ndarray, action_mask: np.ndarray
    ) -> tuple[int, float]:
        probabilities = self.probabilities(features, action_mask)
        action = int(self._rng.choice(self.action_count, p=probabilities))
        return action, log(
            max(float(probabilities[action]), PROBABILITY_EPSILON)
        )

    def make_candidate(
        self, features: np.ndarray, target_probabilities: np.ndarray
    ) -> DecisionTreeRegressor:
        expected_shape = (features.shape[0], self.action_count)

        if target_probabilities.shape != expected_shape:
            raise ValueError(
                f"target_probabilities has shape {target_probabilities.shape}, "
                f"expected {expected_shape}"
            )

        candidate = DecisionTreeRegressor(
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.seed + self._fit_count,
        )
        candidate.fit(features, target_probabilities)

        self._fit_count += 1
        return candidate

    def replace_tree(self, tree: DecisionTreeRegressor) -> None:
        self.tree = tree

    @property
    def depth(self) -> int:
        if self.tree is None:
            return 0
        return int(self.tree.get_depth())

    @property
    def leaf_count(self) -> int:
        if self.tree is None:
            return 1
        return int(self.tree.get_n_leaves())
