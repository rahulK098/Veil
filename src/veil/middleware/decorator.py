"""``@redacted_call``: wrap an LLM client call with inbound and outbound scrubbing."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast, overload

from veil.middleware.adapters import CallAdapter, TextAdapter
from veil.middleware.guard import Guard

RequestIdFn = Callable[..., str | None]


@overload
def redacted_call[**P, R](
    guard: Guard,
    *,
    adapter: CallAdapter | None = ...,
    request_id: RequestIdFn | None = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def redacted_call[**P, R](
    guard: Guard,
    fn: Callable[P, R],
    *,
    adapter: CallAdapter | None = ...,
    request_id: RequestIdFn | None = ...,
) -> Callable[P, R]: ...


def redacted_call[**P, R](
    guard: Guard,
    fn: Callable[P, R] | None = None,
    *,
    adapter: CallAdapter | None = None,
    request_id: RequestIdFn | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Scrub inputs on the way in and outputs on the way out.

    ``adapter`` locates text in the call (default: first positional ``str`` argument and a
    ``str`` return value). ``request_id`` is an optional ``fn(*args, **kwargs) -> str`` used to
    correlate the inbound and outbound audit records. Works for sync and ``async def``.

    Usage::

        @redacted_call(guard, adapter=ChatMessagesAdapter())
        def chat(*, messages): return client.chat.completions.create(model=..., messages=messages)
    """
    chosen = adapter or TextAdapter()

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):
            async_func = cast(Callable[P, Awaitable[Any]], func)

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                rid = request_id(*args, **kwargs) if request_id else None
                new_args, new_kwargs = _scrub_inputs(guard, chosen, args, kwargs, rid)
                result = await async_func(*new_args, **new_kwargs)
                return _scrub_outputs(guard, chosen, result, rid)

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            rid = request_id(*args, **kwargs) if request_id else None
            new_args, new_kwargs = _scrub_inputs(guard, chosen, args, kwargs, rid)
            result = func(*new_args, **new_kwargs)
            return cast(R, _scrub_outputs(guard, chosen, result, rid))

        return wrapper

    return decorate(fn) if fn is not None else decorate


def _scrub_inputs(
    guard: Guard,
    adapter: CallAdapter,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    rid: str | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    texts = adapter.extract_inputs(args, kwargs)
    scrubbed = [guard.inbound(t, request_id=rid) for t in texts]
    return adapter.replace_inputs(args, kwargs, scrubbed)


def _scrub_outputs(guard: Guard, adapter: CallAdapter, result: Any, rid: str | None) -> Any:
    texts = adapter.extract_outputs(result)
    if not texts:
        return result
    scrubbed = [guard.outbound(t, request_id=rid) for t in texts]
    return adapter.replace_outputs(result, scrubbed)
