"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts
# 원하는 기본 모델 이름을 정의합니다. Configuration 클래스의 현재 기본값을 대체하기 위한 것입니다.
# 참고: Configuration 클래스 내 'model' 필드의 기본값을 수동으로 업데이트해야 합니다.
NEW_DEFAULT_MODEL = "google_genai/gemini-2.5-pro-exp-03-25"

# 모델 초기화 시 비동기 옵션 사용 여부에 대한 기본값을 정의합니다.
# 이 값은 `Configuration` 클래스 내에 해당 옵션 필드가 정의될 경우,
# 그 필드의 기본값으로 사용될 수 있습니다.
# 예를 들어, `Configuration` 클래스에 `model_async_init: bool = field(default=DEFAULT_MODEL_ASYNC_INIT)`와
# 같이 필드가 추가될 때 이 상수가 사용될 수 있습니다.
# True로 설정하면, 해당 모델 또는 SDK가 지원하는 경우 비동기 방식으로 모델 클라이언트를
# 초기화하도록 시도할 수 있습니다.
DEFAULT_MODEL_ASYNC_INIT: bool = True

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default=NEW_DEFAULT_MODEL,
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    streaming: bool = field(
        default=DEFAULT_MODEL_ASYNC_INIT,
        metadata={
            "description": "Whether to stream the agent's responses. "
            "When set to True, responses will be streamed as they are generated."
        },

    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
