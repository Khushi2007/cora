from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from utils import clean_tool_output


# --------------------------------------------------
# Wikipedia
# --------------------------------------------------

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=2000,
    )
)


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for reliable encyclopaedic information.

    Use this tool for:
    - biographies
    - historical events
    - scientific concepts
    - definitions
    - countries
    - organizations
    - literature
    - mathematics
    - technology
    - well-established factual topics

    Do NOT use this tool for:
    - breaking news
    - current events
    - live information
    - rapidly changing facts

    Args:
        query: The topic or person to search for on Wikipedia.
    """

    try:
        result = wiki.run(query)

        result = clean_tool_output(result)

        if len(result) > 2000:
            result = result[:2000] + "\n..."
        
        return result
    
    except Exception:
        return """
STATUS: FAILED

Wikipedia search is unavailable.

This is likely temporary.

Do not retry this search.
Use another tool instead.
"""