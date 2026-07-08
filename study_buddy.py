import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from tools_vision import analyze_image_paths

load_dotenv()


study_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)


MATH_FORMATTING_PROMPT = (
    " Format the response in clean Markdown. For scientific and mathematical content, "
    "write equations, formulas, variables, units, derivatives, integrals, matrices, "
    "vectors, chemical notation, and scientific notation using LaTeX. Use inline math "
    "with \\(...\\) for short expressions and block math with \\[...\\] for important "
    "formulas or multi-step derivations. Use Markdown tables where comparison helps. "
    "Avoid plain-text approximations such as x^2, sqrt(x), or a/b when LaTeX would be clearer."
)


STUDY_PROMPTS = {
    "Explain Topic": (
        "You are CORA Study Buddy, a clear and patient tutor. Teach the concept step by step, "
        "use examples, call out common confusions, and end with a concise summary."
        + MATH_FORMATTING_PROMPT
    ),
    "Summarize Notes": (
        "You are CORA Study Buddy, an expert note organizer. Convert the material into organized, "
        "concise notes with headings, key ideas, definitions, examples, and important details."
        + MATH_FORMATTING_PROMPT
    ),
    "Generate Quiz": (
        "You are CORA Study Buddy, a quiz generator. Create MCQs and short-answer questions. "
        "Put answers in a separate answer key after the questions so learners can try first."
        + MATH_FORMATTING_PROMPT
    ),
    "Generate Flashcards": (
        "You are CORA Study Buddy, a flashcard creator. Produce clear Question and Answer pairs "
        "covering definitions, concepts, formulas, examples, and likely exam points."
        + MATH_FORMATTING_PROMPT
    ),
    "Exam Practice": (
        "You are CORA Study Buddy, an exam coach. Generate 2 mark, 5 mark, 10 mark, and long-answer "
        "questions with concise guidance or model-answer outlines."
        + MATH_FORMATTING_PROMPT
    ),
    "Revision Sheet": (
        "You are CORA Study Buddy, a revision-sheet specialist. Compress the material into a quick "
        "revision sheet highlighting formulas, definitions, important points, and likely mistakes."
        + MATH_FORMATTING_PROMPT
    ),
}


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


def noop_progress(value, desc=None):
    return None


def _file_path(file_obj) -> str:
    if isinstance(file_obj, str):
        return file_obj

    if hasattr(file_obj, "name"):
        return file_obj.name

    raise ValueError("Unsupported uploaded file format.")


def _extract_pdf_pages(pdf_path: str) -> list[dict]:
    pages = PyPDFLoader(pdf_path).load()
    extracted = []

    for index, page in enumerate(pages):
        text = (page.page_content or "").strip()
        page_number = page.metadata.get("page", index) + 1

        extracted.append(
            {
                "page": page_number,
                "text": text,
                "needs_vision": len(text) < 80,
            }
        )

    return extracted


def _render_pdf_pages(pdf_path: str, page_numbers: list[int], output_dir: str) -> list[str]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ValueError(
            "Scanned or handwritten PDF pages require pypdfium2. "
            "Install the updated requirements and try again."
        ) from exc

    rendered_paths = []
    pdf = pdfium.PdfDocument(pdf_path)

    for page_number in page_numbers:
        page = pdf[page_number - 1]
        bitmap = page.render(scale=2.0)
        image = bitmap.to_pil()
        output_path = os.path.join(
            output_dir,
            f"{Path(pdf_path).stem}_page_{page_number}.png",
        )
        image.save(output_path)
        rendered_paths.append(output_path)

    return rendered_paths


def _extract_text_from_images(paths: list[str], source_label: str) -> str:
    prompt = (
        "Extract all readable study material from these images. "
        "Include typed text, handwriting, formulas, labels, tables, diagrams, and whiteboard content. "
        "Preserve order as much as possible. If a diagram is important, describe it in words. "
        "When you extract formulas, equations, variables, matrices, chemical notation, or scientific "
        "symbols, preserve them as LaTeX using \\(...\\) for inline math and \\[...\\] for block equations."
    )

    extracted = analyze_image_paths(paths, prompt)

    return f"### Visual extraction from {source_label}\n\n{extracted}".strip()


def _extract_study_material(uploaded_files, progress=noop_progress) -> str:
    if not uploaded_files:
        return ""

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    sections = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file_obj in uploaded_files:
            path = _file_path(file_obj)
            suffix = Path(path).suffix.lower()
            filename = os.path.basename(path)

            if suffix == ".pdf":
                progress(0.25, desc=f"Extracting text from {filename}...")
                pages = _extract_pdf_pages(path)

                typed_sections = []
                pages_for_vision = []

                for page in pages:
                    if page["text"]:
                        typed_sections.append(
                            f"Page {page['page']}:\n{page['text']}"
                        )

                    if page["needs_vision"]:
                        pages_for_vision.append(page["page"])

                if typed_sections:
                    sections.append(
                        f"### Text extracted from {filename}\n\n"
                        + "\n\n".join(typed_sections)
                    )

                if pages_for_vision:
                    progress(0.45, desc=f"Analyzing scanned pages in {filename}...")
                    rendered = _render_pdf_pages(path, pages_for_vision, temp_dir)
                    sections.append(
                        _extract_text_from_images(
                            rendered,
                            f"{filename} pages {', '.join(map(str, pages_for_vision))}",
                        )
                    )

            elif suffix in IMAGE_EXTENSIONS:
                progress(0.45, desc=f"Analyzing handwritten notes in {filename}...")
                sections.append(_extract_text_from_images([path], filename))

            else:
                raise ValueError(
                    "Study Buddy supports PDF files and common image formats."
                )

    return "\n\n---\n\n".join(section for section in sections if section.strip())


def run_study_task(
    task: str,
    text_input: str,
    uploaded_files,
    progress=noop_progress,
) -> str:
    """
    Runs a focused Study Buddy workflow over typed input and uploaded study material.
    """

    system_prompt = STUDY_PROMPTS.get(task)

    if system_prompt is None:
        raise ValueError("Please select a valid Study Buddy task.")

    typed_text = text_input.strip() if text_input else ""

    progress(0.10, desc="Preparing study material...")
    extracted_material = _extract_study_material(uploaded_files, progress)

    if not typed_text and not extracted_material:
        raise ValueError("Please enter a topic, paste notes, or upload study material.")

    progress(0.75, desc="Creating learning resource...")

    user_prompt = f"""
Task: {task}

Topic, question, or pasted notes:
{typed_text if typed_text else "None provided."}

Uploaded study material:
{extracted_material if extracted_material else "None provided."}
"""

    response = study_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    progress(1.0, desc="Done!")

    return response.content
