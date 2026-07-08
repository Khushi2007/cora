from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from utils import clean_tool_output

# --------------------------------------------------
# DuckDuckGo
# --------------------------------------------------

duckduckgo = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """
    Search the live web for recent or time-sensitive information.

    Use this tool for:
    - current events
    - breaking news
    - recent research
    - product releases
    - company updates
    - sports
    - weather
    - stock prices
    - information that changes frequently
    - events occurring after 2024

    Do NOT use this tool for:
    - biographies
    - historical events
    - scientific concepts
    - definitions
    - general background knowledge

    Args:
        query: The search query to look up on the web.
    """

    try:
        result = duckduckgo.run(query)

        result = clean_tool_output(result)

        if len(result) > 2000:
            result = result[:2000] + "\n..."

        return result

    except Exception as e:
        return f"Web search failed:\n{e}"