"""BFCL handler for Qwen3.8-27B served by our own vLLM (OpenAI-compatible API), thinking on.

Built on BFCL's QwenAPIHandler (the official handler for Qwen3 "-FC" thinking models, which talks to
Alibaba's DashScope API with streaming). Differences:
- the client points at the local vLLM server (QWEN38_BASE_URL, default http://127.0.0.1:8000/v1);
- thinking is switched on through vLLM's chat_template_kwargs instead of DashScope's enable_thinking;
- temperature is passed (BFCL default 0.001; QwenAPIHandler leaves it to the server);
- vLLM streams reasoning as `delta.reasoning`, DashScope as `delta.reasoning_content`; both are accepted.

Tool calls come back already parsed by vLLM (--tool-call-parser qwen3_coder), so BFCL scores them in FC mode.
install_handler.py copies this file into bfcl_eval/model_handler/api_inference/ and registers the model.
"""

import os
from typing import Any

from bfcl_eval.model_handler.api_inference.qwen import QwenAPIHandler
from openai import BadRequestError, OpenAI
from overrides import override

MAX_TOKENS = int(os.getenv("QWEN38_MAX_TOKENS", "16384"))   # 0 = no cap

# The parent handlers build OpenAI/DashScope clients in __init__ and those refuse to start without a key.
# They're replaced by the local client below and never used, so placeholder keys are enough.
os.environ.setdefault("OPENAI_API_KEY", "unused-local-vllm")
os.environ.setdefault("QWEN_API_KEY", "unused-local-vllm")


class Qwen38VLLMHandler(QwenAPIHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.client = OpenAI(
            base_url=os.getenv("QWEN38_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("QWEN38_API_KEY", "EMPTY"),
            timeout=float(os.getenv("QWEN38_TIMEOUT", "1800")),
        )

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = dict(
            messages=message,
            model=self.model_name.replace("-FC", ""),
            tools=tools,
            parallel_tool_calls=True,
            temperature=self.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            stream=True,
            stream_options={"include_usage": True},
        )
        # Cap one model call's output (thinking included). At BFCL's near-greedy temperature the model sometimes
        # repeats itself until the context is full (one simple_java entry ran 13+ min); real answers here use
        # a few hundred to a few thousand tokens, and a looping answer fails either way.
        if MAX_TOKENS:
            kwargs["max_tokens"] = MAX_TOKENS
        # 4 live_irrelevance entries (120-9-0 .. 123-9-3) come with no tools at all. vLLM rejects `tools: []`
        # ("must not be an empty array... or omit the field"), and BFCL would score the resulting error text as a
        # correct "no call", so leave the tool fields out and let the model just answer.
        if not tools:
            kwargs.pop("tools"); kwargs.pop("parallel_tool_calls")
        try:
            return self.generate_with_backoff(**kwargs)
        except BadRequestError as e:
            # prompt + max_tokens over vLLM's --max-model-len: retry without the cap (only very long prompts)
            if "max_tokens" in kwargs and "context" in str(e).lower():
                kwargs.pop("max_tokens")
                return self.generate_with_backoff(**kwargs)
            raise

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        def with_reasoning_content(stream):
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning", None)
                    if reasoning and not getattr(delta, "reasoning_content", None):
                        delta.reasoning_content = reasoning
                yield chunk

        return super()._parse_query_response_FC(with_reasoning_content(api_response))
