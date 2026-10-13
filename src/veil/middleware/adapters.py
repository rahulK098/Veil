"""Call adapters: how ``@redacted_call`` finds text inside a call's arguments and result.

An adapter is the only thing that knows the shape of a particular client's API, so wrapping
a new provider means writing one small class, not touching the guard.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any, Protocol

Args = tuple[Any, ...]
Kwargs = dict[str, Any]


class CallAdapter(Protocol):
    def extract_inputs(self, args: Args, kwargs: Kwargs) -> list[str]: ...

    def replace_inputs(
        self, args: Args, kwargs: Kwargs, texts: Sequence[str]
    ) -> tuple[Args, Kwargs]: ...

    def extract_outputs(self, result: Any) -> list[str]: ...

    def replace_outputs(self, result: Any, texts: Sequence[str]) -> Any: ...


class TextAdapter:
    """For ``fn(prompt: str, ...) -> str``. ``arg`` is a positional index or keyword name."""

    def __init__(self, arg: int | str = 0) -> None:
        self._arg = arg

    def extract_inputs(self, args: Args, kwargs: Kwargs) -> list[str]:
        value = args[self._arg] if isinstance(self._arg, int) else kwargs[self._arg]
        if not isinstance(value, str):
            raise TypeError(f"expected str prompt at {self._arg!r}, got {type(value).__name__}")
        return [value]

    def replace_inputs(
        self, args: Args, kwargs: Kwargs, texts: Sequence[str]
    ) -> tuple[Args, Kwargs]:
        (text,) = texts
        if isinstance(self._arg, int):
            new_args = tuple(text if i == self._arg else a for i, a in enumerate(args))
            return new_args, dict(kwargs)
        return args, {**kwargs, self._arg: text}

    def extract_outputs(self, result: Any) -> list[str]:
        return [result] if isinstance(result, str) else []

    def replace_outputs(self, result: Any, texts: Sequence[str]) -> Any:
        return texts[0] if texts and isinstance(result, str) else result


class ChatMessagesAdapter:
    """For OpenAI / Anthropic style ``fn(*, messages=[{"role", "content"}, ...])``.

    Every string ``content`` in ``messages`` is scrubbed. Output extraction tries, in order:
    OpenAI ``result.choices[i].message.content``, Anthropic ``result.content[i].text``,
    and a plain ``str``. Objects are copied before mutation so the provider's original
    response is left intact.
    """

    def __init__(self, kwarg: str = "messages") -> None:
        self._kwarg = kwarg

    def extract_inputs(self, args: Args, kwargs: Kwargs) -> list[str]:
        messages = kwargs.get(self._kwarg)
        if not isinstance(messages, list):
            raise TypeError(f"expected a list at kwarg {self._kwarg!r}")
        return [m["content"] for m in messages if isinstance(m.get("content"), str)]

    def replace_inputs(
        self, args: Args, kwargs: Kwargs, texts: Sequence[str]
    ) -> tuple[Args, Kwargs]:
        replacements = iter(texts)
        new_messages = [
            {**m, "content": next(replacements)} if isinstance(m.get("content"), str) else dict(m)
            for m in kwargs[self._kwarg]
        ]
        return args, {**kwargs, self._kwarg: new_messages}

    def extract_outputs(self, result: Any) -> list[str]:
        return [text for _, text in _output_slots(result)]

    def replace_outputs(self, result: Any, texts: Sequence[str]) -> Any:
        if isinstance(result, str):
            return texts[0] if texts else result
        clone = copy.deepcopy(result)
        for (setter, _), text in zip(_output_slots(clone), texts, strict=True):
            setter(text)
        return clone


def _output_slots(result: Any) -> list[tuple[Callable[[str], None], str]]:
    """Return (setter, current_text) pairs for every text field in a provider response."""
    if isinstance(result, str):
        return [(lambda _t: None, result)]
    slots: list[tuple[Callable[[str], None], str]] = []
    choices = getattr(result, "choices", None)
    if choices is not None:  # OpenAI chat completion
        for choice in choices:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                slots.append((_attr_setter(message, "content"), content))
        return slots
    content_blocks = getattr(result, "content", None)
    if isinstance(content_blocks, list):  # Anthropic message
        for block in content_blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                slots.append((_attr_setter(block, "text"), text))
    return slots


def _attr_setter(obj: Any, name: str) -> Callable[[str], None]:
    def _set(value: str) -> None:
        setattr(obj, name, value)

    return _set
