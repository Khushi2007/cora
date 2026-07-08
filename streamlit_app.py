import os
import re
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError


load_dotenv()

st.set_page_config(
    page_title="CORA",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to your .env file or deployment secrets.")
    st.stop()

from agent import run_agent_with_trace  # noqa: E402
from developer_studio import run_developer_task  # noqa: E402
from study_buddy import run_study_task  # noqa: E402
from tools_rag import clear_documents, index_documents, search_documents  # noqa: E402
import tools_vision  # noqa: E402


UPLOAD_ROOT = Path(".streamlit_uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)

PDF_TYPES = ["pdf"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"]


def init_state():
    defaults = {
        "messages": [],
        "reasoning_trace": "Reasoning will appear here after your first message.",
        "session_id": str(uuid.uuid4()),
        "uploaded_pdfs": [],
        "uploaded_images": [],
        "pdf_status": "No documents indexed.",
        "image_status": "No images uploaded.",
        "image_analysis": "",
        "doc_answer": "",
        "dev_status": "Ready.",
        "dev_output": "",
        "study_status": "Ready.",
        "study_output": "",
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clean_name(name: str) -> str:
    stem = Path(name).stem or "upload"
    suffix = Path(name).suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return f"{safe_stem or 'upload'}{suffix}"


def save_uploaded_files(files, folder: str) -> list[str]:
    if not files:
        return []

    target_dir = UPLOAD_ROOT / st.session_state.session_id / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file in files:
        path = target_dir / clean_name(file.name)
        path.write_bytes(file.getbuffer())
        saved_paths.append(str(path))

    return saved_paths


def make_progress(status_slot):
    progress_bar = st.progress(0)

    def update(value, desc=None):
        value = max(0.0, min(float(value), 1.0))
        progress_bar.progress(value)
        if desc:
            status_slot.info(desc)

    return update


def friendly_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()

    if isinstance(exc, GraphRecursionError):
        return "The agent exceeded its reasoning limit. Try asking a simpler or more specific question."
    if "rate_limit" in lowered:
        return "Groq rate limit reached. Please wait a few seconds and try again."
    if "api key" in lowered or "authentication" in lowered:
        return "Invalid or missing GROQ_API_KEY."
    if isinstance(exc, FileNotFoundError):
        return "The selected file could not be found. Please upload it again."
    if isinstance(exc, ValueError):
        return message
    if "connection" in lowered or "network" in lowered:
        return "Unable to reach the model provider. Please check your connection and try again."

    return "Sorry, something unexpected happened while processing your request."


def reset_chat():
    st.session_state.messages = []
    st.session_state.reasoning_trace = "Reasoning will appear here after your first message."
    st.session_state.session_id = str(uuid.uuid4())


def clear_pdfs():
    clear_documents()
    st.session_state.uploaded_pdfs = []
    st.session_state.pdf_status = "No documents indexed."
    st.session_state.doc_answer = ""


def clear_images():
    tools_vision.clear_image()
    st.session_state.uploaded_images = []
    st.session_state.image_status = "No images uploaded."
    st.session_state.image_analysis = ""


def render_sidebar():
    with st.sidebar:
        st.title("CORA")
        st.caption("Cognitive Omnitask Reasoning Assistant")
        st.write(
            "A multimodal assistant for documents, images, web-aware reasoning, "
            "developer workflows, and study support."
        )

        st.divider()
        st.subheader("Session")
        st.code(st.session_state.session_id, language=None)

        if st.button("New Chat", use_container_width=True):
            reset_chat()
            st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear Uploaded PDFs", use_container_width=True):
                clear_pdfs()
                st.rerun()
        with col_b:
            if st.button("Clear Uploaded Images", use_container_width=True):
                clear_images()
                st.rerun()

        st.divider()
        st.subheader("Uploaded PDFs")
        if st.session_state.uploaded_pdfs:
            for path in st.session_state.uploaded_pdfs:
                st.caption(f"- {Path(path).name}")
        else:
            st.caption("No PDFs uploaded.")

        st.subheader("Uploaded Images")
        if st.session_state.uploaded_images:
            for path in st.session_state.uploaded_images:
                st.caption(f"- {Path(path).name}")
        else:
            st.caption("No images uploaded.")

        st.divider()
        st.caption("Built with Streamlit, LangGraph, Groq, ChromaDB, LangChain, and HuggingFace embeddings.")


def render_welcome() -> str | None:
    st.markdown(
        """
        <div style="text-align:center; padding:2.5rem 1rem 1.5rem;">
            <h1>Welcome to CORA</h1>
            <h4>Cognitive Omnitask Reasoning Assistant</h4>
            <p>Reason across documents, images, the web, and conversation history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prompts = {
        "📄 Summarize PDF": "Summarize my uploaded PDF.",
        "🖼 Compare Images": "Compare these uploaded images.",
        "🌐 AI News": "What's happening in AI today?",
        "📊 Explain Graph": "Explain this graph.",
    }

    cols = st.columns(4)
    selected = None
    for col, (label, prompt) in zip(cols, prompts.items()):
        with col:
            if st.button(label, use_container_width=True):
                selected = prompt

    return selected


def stream_chat(prompt: str, trace_slot):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    assistant_message = "🧠 CORA is thinking..."
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    with st.chat_message("assistant"):
        response_slot = st.empty()
        response_slot.markdown(assistant_message)
        start = time.perf_counter()

        try:
            for event in run_agent_with_trace(prompt, st.session_state.session_id):
                event_type = event.get("type")

                if event.get("trace"):
                    st.session_state.reasoning_trace = event["trace"]
                    trace_slot.code(st.session_state.reasoning_trace, language=None)

                if event_type == "status":
                    assistant_message = event.get("message", assistant_message)
                    response_slot.markdown(assistant_message)
                elif event_type == "token":
                    assistant_message = event.get("text", assistant_message)
                    response_slot.markdown(assistant_message)
                elif event_type == "done":
                    elapsed = time.perf_counter() - start
                    assistant_message = (
                        event.get("text", assistant_message)
                        + f"\n\n---\n⏱ Response generated in {elapsed:.2f} seconds"
                    )
                    response_slot.markdown(assistant_message)
        except Exception as exc:
            assistant_message = friendly_error(exc)
            response_slot.error(assistant_message)

    st.session_state.messages[-1]["content"] = assistant_message


def hybrid_chat_tab():
    chat_col, trace_col = st.columns([0.68, 0.32], gap="large")

    with trace_col:
        with st.expander("🔍 CORA Reasoning Trace", expanded=False):
            trace_slot = st.empty()
            trace_slot.code(st.session_state.reasoning_trace, language=None)

    with chat_col:
        selected_prompt = None
        if not st.session_state.messages:
            selected_prompt = render_welcome()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        typed_prompt = st.chat_input("Ask CORA anything...")
        prompt = selected_prompt or typed_prompt

        if prompt:
            stream_chat(prompt, trace_slot)


def document_qa_tab():
    st.subheader("Document QA")
    st.caption("Upload PDFs, index them into ChromaDB, then ask questions over the indexed documents.")

    uploaded = st.file_uploader(
        "Upload PDF files",
        type=PDF_TYPES,
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    status_slot = st.empty()
    status_slot.info(st.session_state.pdf_status)

    col_a, col_b = st.columns([0.5, 0.5])
    with col_a:
        if st.button("Index Uploaded PDFs", type="primary", use_container_width=True):
            if not uploaded:
                st.warning("Upload one or more PDFs first.")
            else:
                try:
                    paths = save_uploaded_files(uploaded, "pdfs")
                    st.session_state.uploaded_pdfs = paths
                    progress = make_progress(status_slot)
                    st.session_state.pdf_status = index_documents(paths, progress)
                    status_slot.success(st.session_state.pdf_status)
                except Exception as exc:
                    st.error(friendly_error(exc))
    with col_b:
        if st.button("Clear PDFs", use_container_width=True):
            clear_pdfs()
            st.rerun()

    if st.session_state.uploaded_pdfs:
        st.markdown("**Indexed documents**")
        for path in st.session_state.uploaded_pdfs:
            st.write(f"- {Path(path).name}")

    question = st.text_input("Ask a question about the indexed PDFs", placeholder="What are the main findings?")
    if st.button("Search Documents", use_container_width=True):
        if not question.strip():
            st.warning("Enter a document question first.")
        else:
            try:
                st.session_state.doc_answer = search_documents.invoke({"query": question})
            except Exception as exc:
                st.session_state.doc_answer = friendly_error(exc)

    if st.session_state.doc_answer:
        st.markdown("**Answer**")
        st.markdown(st.session_state.doc_answer)


def image_studio_tab():
    st.subheader("Image Studio")
    st.caption("Upload multiple images, preview them, and analyze or compare them with CORA Vision.")

    uploaded = st.file_uploader(
        "Upload image files",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key="image_uploader",
    )

    col_a, col_b = st.columns([0.5, 0.5])
    with col_a:
        if st.button("Save Uploaded Images", type="primary", use_container_width=True):
            if not uploaded:
                st.warning("Upload one or more images first.")
            else:
                try:
                    paths = save_uploaded_files(uploaded, "images")
                    st.session_state.uploaded_images = paths
                    tools_vision.set_images(paths)
                    count = len(paths)
                    st.session_state.image_status = f"{count} image{'s' if count != 1 else ''} uploaded."
                    st.success(st.session_state.image_status)
                except Exception as exc:
                    st.error(friendly_error(exc))
    with col_b:
        if st.button("Remove Images", use_container_width=True):
            clear_images()
            st.rerun()

    st.info(st.session_state.image_status)

    if st.session_state.uploaded_images:
        st.markdown("**Uploaded images**")
        preview_cols = st.columns(min(3, len(st.session_state.uploaded_images)))
        for index, path in enumerate(st.session_state.uploaded_images):
            with preview_cols[index % len(preview_cols)]:
                st.image(path, caption=Path(path).name, use_container_width=True)

    analysis_prompt = st.text_area(
        "Analysis prompt",
        value=(
            "Analyze all uploaded images.\n\n"
            "For each image, describe it in detail.\n"
            "Extract any visible text.\n"
            "If there are multiple images, compare them and explain similarities, differences, or relationships."
        ),
        height=150,
    )

    if st.button("Analyze Images", type="primary", use_container_width=True):
        with st.spinner("Analyzing images..."):
            try:
                st.session_state.image_analysis = tools_vision.describe_image.invoke(
                    {"prompt": analysis_prompt}
                )
            except Exception as exc:
                st.session_state.image_analysis = friendly_error(exc)

    if st.session_state.image_analysis:
        st.markdown("**Vision analysis**")
        st.markdown(st.session_state.image_analysis)


def developer_studio_tab():
    st.subheader("Developer Studio")
    st.caption("Understand, debug, optimize, generate, translate, and analyze code.")

    col_a, col_b = st.columns(2)
    with col_a:
        language = st.selectbox(
            "Language",
            ["Auto Detect", "Python", "C++", "Java", "JavaScript", "C#", "Go", "Rust"],
        )
    with col_b:
        task = st.selectbox(
            "Task",
            [
                "Explain Code",
                "Debug Code",
                "Optimize Code",
                "Complexity Analysis",
                "Language Conversion",
                "Generate Code",
                "Generate Test Cases",
            ],
        )

    code_input = st.text_area(
        "Code Input or Program Description",
        placeholder="Paste code here, or describe the program you want to generate...",
        height=300,
    )
    instructions = st.text_area(
        "Optional Additional Instructions, Error Message, or Target Language",
        placeholder="Add constraints, expected behavior, stack trace, compiler error, target language, or style preferences...",
        height=120,
    )

    if st.button("Run Developer Task", type="primary", use_container_width=True):
        with st.spinner("Working on your code..."):
            try:
                st.session_state.dev_output = run_developer_task(
                    language,
                    task,
                    code_input,
                    instructions,
                )
                st.session_state.dev_status = "Done."
            except Exception as exc:
                st.session_state.dev_status = friendly_error(exc)
                st.session_state.dev_output = ""

    st.info(st.session_state.dev_status)
    if st.session_state.dev_output:
        st.markdown(st.session_state.dev_output)


def study_buddy_tab():
    st.subheader("Study Buddy")
    st.caption("Turn notes, PDFs, scanned pages, and handwritten images into learning resources.")

    task = st.selectbox(
        "Task",
        [
            "Explain Topic",
            "Summarize Notes",
            "Generate Quiz",
            "Generate Flashcards",
            "Exam Practice",
            "Revision Sheet",
        ],
    )

    topic_notes = st.text_area(
        "Topic / Question / Notes",
        placeholder="Enter a topic, ask a study question, or paste notes...",
        height=240,
    )

    uploaded = st.file_uploader(
        "Optional Study Material",
        type=PDF_TYPES + IMAGE_TYPES,
        accept_multiple_files=True,
        key="study_uploader",
    )

    status_slot = st.empty()
    status_slot.info(st.session_state.study_status)

    if st.button("Run Study Task", type="primary", use_container_width=True):
        try:
            paths = save_uploaded_files(uploaded, "study") if uploaded else []
            progress = make_progress(status_slot)
            with st.spinner("Creating learning resource..."):
                st.session_state.study_output = run_study_task(
                    task,
                    topic_notes,
                    paths,
                    progress,
                )
            st.session_state.study_status = "Done."
            status_slot.success("Done.")
        except Exception as exc:
            st.session_state.study_status = friendly_error(exc)
            status_slot.error(st.session_state.study_status)
            st.session_state.study_output = ""

    if st.session_state.study_output:
        st.markdown(st.session_state.study_output)


def main():
    init_state()
    render_sidebar()

    st.title("CORA")
    st.caption("Cognitive Omnitask Reasoning Assistant")

    tab_chat, tab_docs, tab_images, tab_dev, tab_study = st.tabs(
        [
            "💬 Hybrid Chat",
            "📄 Document QA",
            "🖼 Image Studio",
            "💻 Developer Studio",
            "🎓 Study Buddy",
        ]
    )

    with tab_chat:
        hybrid_chat_tab()
    with tab_docs:
        document_qa_tab()
    with tab_images:
        image_studio_tab()
    with tab_dev:
        developer_studio_tab()
    with tab_study:
        study_buddy_tab()


if __name__ == "__main__":
    main()
