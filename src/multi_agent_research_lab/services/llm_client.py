"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Keep retry, timeout, and token logging here rather than inside agents.
        """

        if not self.settings.openai_api_key:
            raise AgentExecutionError("OPENAI_API_KEY is missing. Add it to .env first.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentExecutionError(
                'OpenAI SDK is not installed. Run: pip install -e ".[llm]"'
            ) from exc

        client = OpenAI(api_key=self.settings.openai_api_key, timeout=self.settings.timeout_seconds)
        response = self._create_completion(client, system_prompt, user_prompt)
        content = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            content=content.strip(),
            input_tokens=None if usage is None else usage.prompt_tokens,
            output_tokens=None if usage is None else usage.completion_tokens,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _create_completion(self, client: Any, system_prompt: str, user_prompt: str) -> Any:
        """Call the provider API with retry for transient failures."""

        try:
            return client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:
            raise AgentExecutionError(f"LLM completion failed: {exc}") from exc
