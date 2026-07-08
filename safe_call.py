import functools

import gradio as gr
import groq

from langgraph.errors import GraphRecursionError


def _convert_exception(e):

    if isinstance(e, GraphRecursionError):
        raise gr.Error(
            "⚠️ The agent exceeded its reasoning limit.\n\n"
            "Try asking a simpler question."
        )

    elif isinstance(e, groq.RateLimitError):
        raise gr.Error(
            "⏱️ Groq rate limit reached.\n\n"
            "Please wait a few seconds and try again."
        )

    elif isinstance(e, groq.APIConnectionError):
        raise gr.Error(
            "🌐 Unable to reach Groq.\n\n"
            "Please check your internet connection."
        )

    elif isinstance(e, groq.AuthenticationError):
        raise gr.Error(
            "🔑 Invalid GROQ_API_KEY."
        )

    elif isinstance(e, FileNotFoundError):
        raise gr.Error(
            "📄 The selected file could not be found."
        )

    elif isinstance(e, ValueError):
        raise gr.Error(str(e))

    else:
        raise gr.Error(f"{type(e).__name__}: {e}")
    

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