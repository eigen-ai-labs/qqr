from abc import ABC, abstractmethod


class LLMJudge(ABC):
    @abstractmethod
    async def compare(
        self, messages_a: list[dict], messages_b: list[dict], *args, **kwargs
    ) -> tuple[float, float]: ...

    @abstractmethod
    async def bidirectional_compare(
        self, messages_a: list[dict], messages_b: list[dict], *args, **kwargs
    ) -> tuple[float, float, dict]: ...
