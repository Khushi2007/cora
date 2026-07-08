import re

# --------------------------------------------------
# Clean-up Result
# --------------------------------------------------

def clean_tool_output(text: str) -> str:

    text = re.sub(r"【.*?】", "", text)
    text = re.sub(r"\[\d+†L\d+(?:-L?\d+)?\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()