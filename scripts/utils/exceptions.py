import asyncio


class RolloutCircuitOpen(RuntimeError):
    pass


class FailureCircuitBreaker:
    def __init__(
        self,
        *,
        max_failures: int,
        window_seconds: float,
        cooldown_seconds: float,
    ) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds

        self._failures: list[float] = []
        self._lock = asyncio.Lock()

    async def record_failure(self, error: BaseException) -> None:
        now = asyncio.get_running_loop().time()

        async with self._lock:
            cutoff = now - self.window_seconds
            self._failures = [
                timestamp
                for timestamp in self._failures
                if timestamp >= cutoff
            ]
            self._failures.append(now)

            failure_count = len(self._failures)

        if failure_count >= self.max_failures:
            raise RolloutCircuitOpen(
                f"{failure_count} battle failures within "
                f"{self.window_seconds:.1f}s; last error: "
                f"{type(error).__name__}: {error}"
            )

        await asyncio.sleep(self.cooldown_seconds)
