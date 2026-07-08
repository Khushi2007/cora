import os
import base64
import mimetypes
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()


# --------------------------------------------------
# Vision Model
# --------------------------------------------------

vision_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY"),
)


# --------------------------------------------------
# Image Path
# --------------------------------------------------

current_image_paths = []


def set_images(paths):
    global current_image_paths
    current_image_paths = paths


# --------------------------------------------------
# Clear Image
# --------------------------------------------------

def clear_image():
    global current_image_paths
    current_image_paths = []


def analyze_image_paths(paths: list[str], prompt: str) -> str:
    """
    Analyze explicit image paths without changing Image Studio state.
    """

    if not paths:
        return "No image paths were provided."

    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    for path in paths:

        with open(path, "rb") as f:
            image_bytes = f.read()

        image_base64 = base64.b64encode(image_bytes).decode()

        mime_type, _ = mimetypes.guess_type(path)

        if mime_type is None:
            mime_type = "image/jpeg"

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            }
        )

    message = HumanMessage(
        content=content
    )

    response = vision_llm.invoke([message])

    return response.content


# --------------------------------------------------
# Vision Tool
# --------------------------------------------------

@tool
def describe_image(
    prompt: str = (
        "Analyze every uploaded image.\n\n"
        "For each image:\n"
        "- Describe what is shown.\n"
        "- Mention any important text, objects, charts or people.\n\n"
        "If there are multiple images, compare them and explain any relationships or differences."
    )
) -> str:
    """
    Analyze the uploaded images.

    Use this tool whenever the user asks about an uploaded image,
    photograph, diagram, chart, screenshot, graph, drawing,
    handwritten notes, or any visual content.

    If no image has been uploaded, explain that to the user.
    """

    global current_image_paths

    if not current_image_paths:
        return "No image has been uploaded yet. Please ask the user to upload an image first."
    
    try:
        return analyze_image_paths(current_image_paths, prompt)
    
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise
