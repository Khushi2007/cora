import functools

import groq

from langgraph.errors import GraphRecursionError


class UserFacingError(Exception):
    pass


def _convert_exception(e):
    if isinstance(e, GraphRecursionError):
        raise UserFacingError(
            "The agent exceeded its reasoning limit.\n\n"
            "Try asking a simpler question."
        )

    if isinstance(e, groq.RateLimitError):
        raise UserFacingError(
            "Groq rate limit reached.\n\n"
            "Please wait a few seconds and try again."
        )

    if isinstance(e, groq.APIConnectionError):
        raise UserFacingError(
            "Unable to reach Groq.\n\n"
            "Please check your internet connection."
        )

    if isinstance(e, groq.AuthenticationError):
        raise UserFacingError("Invalid GROQ_API_KEY.")

    if isinstance(e, FileNotFoundError):
        raise UserFacingError("The selected file could not be found.")

    if isinstance(e, ValueError):
        raise UserFacingError(str(e))

    raise UserFacingError(f"{type(e).__name__}: {e}")


def safe_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _convert_exception(e)

    return wrapper


def safe_stream(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            for item in func(*args, **kwargs):
                yield item
        except Exception as e:
            _convert_exception(e)

    return wrapper
