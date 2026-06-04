import asyncio
import logging

from app.config import settings
from app.generation.prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger(__name__)


def _deduplicate_sources(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    sources = []
    for c in chunks:
        if c["url"] not in seen:
            sources.append({"title": c["title"], "url": c["url"]})
            seen.add(c["url"])
    return sources


class _OpenAICompatGenerator:
    def __init__(self):
        self._client = None

    def _client_(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
            )
        return self._client

    async def generate(self, query: str, chunks: list[dict]) -> dict:
        user_msg = build_user_message(query, chunks)
        client = self._client_()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_new_tokens,
        )
        answer = response.choices[0].message.content
        return {"answer": answer, "sources": _deduplicate_sources(chunks)}


class _TransformersGenerator:
    def __init__(self):
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            self._pipe = pipeline(
                "text-generation",
                model=settings.llm_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                max_new_tokens=settings.llm_max_new_tokens,
            )
        return self._pipe

    def _run(self, query: str, chunks: list[dict]) -> dict:
        pipe = self._get_pipe()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(query, chunks)},
        ]
        outputs = pipe(messages)
        answer = outputs[0]["generated_text"][-1]["content"]
        return {"answer": answer, "sources": _deduplicate_sources(chunks)}

    async def generate(self, query: str, chunks: list[dict]) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query, chunks)


def _make_generator():
    if settings.llm_backend == "transformers":
        return _TransformersGenerator()
    return _OpenAICompatGenerator()


_generator = None


async def generate_answer(query: str, chunks: list[dict]) -> dict:
    global _generator
    if _generator is None:
        _generator = _make_generator()
    return await _generator.generate(query, chunks)
