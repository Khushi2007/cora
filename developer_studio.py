import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()


developer_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)


TASK_PROMPTS = {
    "Explain Code": (
        "You are CORA Developer Studio, a senior software engineer who explains code clearly. "
        "Explain the purpose, overall logic, important functions, time complexity, space complexity, "
        "and practical improvements. Be specific and structured."
    ),
    "Debug Code": (
        "You are CORA Developer Studio, a debugging specialist. Identify the bug, explain the cause, "
        "provide corrected code, and explain the fix. If an error message is provided, use it carefully."
    ),
    "Optimize Code": (
        "You are CORA Developer Studio, a performance and readability specialist. Return an improved "
        "implementation, compare time and space complexity, and describe readability improvements."
    ),
    "Generate Code": (
        "You are CORA Developer Studio, a production-quality coding assistant. Generate complete, "
        "readable, maintainable code for the requested program. Include brief usage notes when helpful."
    ),
    "Generate Test Cases": (
        "You are CORA Developer Studio, a test design specialist. Generate normal, boundary, invalid, "
        "and stress test cases. Include expected outcomes and concise reasoning."
    ),
    "Complexity Analysis": (
        "You are CORA Developer Studio, an algorithm analysis specialist. Analyze the code or approach, "
        "identify the dominant operations, explain time complexity, space complexity, bottlenecks, "
        "and practical improvements. Be precise about best, average, and worst cases when relevant."
    ),
    "Language Conversion": (
        "You are CORA Developer Studio, a careful code translation specialist. Convert the provided "
        "code or program description into the requested target language while preserving behavior, "
        "edge cases, and readability. Explain important translation choices briefly."
    ),
}


def run_developer_task(
    language: str,
    task: str,
    code_or_request: str,
    instructions: str,
) -> str:
    """
    Runs a focused Developer Studio workflow using a dedicated text model.
    """

    if not code_or_request or not code_or_request.strip():
        raise ValueError("Please enter code or a program description.")

    system_prompt = TASK_PROMPTS.get(task)

    if system_prompt is None:
        raise ValueError("Please select a valid Developer Studio task.")

    language_hint = language or "Auto Detect"
    extra_instructions = instructions.strip() if instructions else "None"

    user_prompt = f"""
Task: {task}
Language: {language_hint}

Input:
{code_or_request.strip()}

Optional additional instructions or error message:
{extra_instructions}
"""

    response = developer_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return response.content
