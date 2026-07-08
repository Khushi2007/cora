import os
import sys
import uuid
import gradio as gr
import time

from agent import run_agent_with_trace
from tools_rag import (
    index_documents,
    clear_documents,
)
import tools_vision
from developer_studio import run_developer_task
from study_buddy import run_study_task

from safe_call import safe_call, safe_stream


# --------------------------------------------------
# Startup Checks
# --------------------------------------------------

os.makedirs("./chroma_store", exist_ok=True)

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY is not set.")
    print("Add it to your .env file or Hugging Face Spaces Secrets.")
    sys.exit(1)


# --------------------------------------------------
# Upload PDFs
# --------------------------------------------------

@safe_call
def upload_pdf(files, uploaded_pdfs):

    if not files:
        return (
            "No PDFs uploaded.",
            uploaded_pdfs,
            "No PDFs uploaded."
        )

    uploaded_pdfs = [file.name for file in files]

    status = index_documents(uploaded_pdfs)

    display = "\n".join(
        f"✅ {os.path.basename(pdf)}"
        for pdf in uploaded_pdfs
    )

    return (
        status,
        uploaded_pdfs,
        display
    )


# --------------------------------------------------
# Clear PDFs
# --------------------------------------------------

@safe_call
def clear_pdf_list():

    clear_documents()

    return (
        None,
        "No PDFs uploaded.",
        [],
        "No PDFs uploaded."
    )


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

@safe_call
def upload_image(files):

    if not files:
        return (
            "No images uploaded.",
            "No images uploaded."
        )
    
    paths = [file.name for file in files]

    tools_vision.set_images(paths)

    display = "\n".join(
        f"✅ {os.path.basename(path)}"
        for path in paths
    )

    return (
        f"✅ {len(paths)} image"
        f"{'' if len(paths)==1 else 's'} uploaded.",
        display
    )


# --------------------------------------------------
# Remove Image
# --------------------------------------------------

@safe_call
def remove_image():

    tools_vision.clear_image()

    return (
        None,
        "No image uploaded.",
        "No image uploaded."
    )


# --------------------------------------------------
# Chat
# --------------------------------------------------

@safe_stream
def chat(message, history, session_id):

    if not message.strip():
        yield (
            "",
            history,
            "",
            session_id,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(interactive=True, value="➤ Send"),
            gr.update(interactive=True)
        )
        return

    # Add user message
    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    # Placeholder assistant message
    history.append(
        {
            "role": "assistant",
            "content": "🧠 CORA is Thinking..."
        }
    )

    # Show chatbot, hide welcome
    yield (
        "",
        history,
        "⚡ Initializing...",
        session_id,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(interactive=False, value="➤ Send"),
        gr.update(interactive=False)
    )

    start = time.perf_counter()

    # Stream agent events
    for event in run_agent_with_trace(message, session_id):

        event_type = event["type"]

        if event_type == "status":

            history[-1]["content"] = event["message"]

            yield (
                gr.skip(),
                history,
                event["trace"],
                session_id,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(interactive=False, value="➤ Send"),
                gr.update(interactive=False)
            )

        elif event_type == "token":

            history[-1]["content"] = event["text"]

            yield (
                gr.skip(),
                history,
                event["trace"],
                session_id,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(interactive=False, value="➤ Send"),
                gr.update(interactive=False)
            )

        elif event_type == "done":

            elapsed = time.perf_counter() - start

            history[-1]["content"] = (
                event["text"]
                + f"\n\n---\n🕑 Response generated in {elapsed:.2f} seconds"
            )

            yield (
                gr.skip(),
                history,
                event["trace"],
                session_id,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(interactive=True, value="➤ Send"),
                gr.update(interactive=True)
            )


# --------------------------------------------------
# Analyze Uploaded Images
# --------------------------------------------------

@safe_call
def analyze_uploaded_images():

    return tools_vision.describe_image.invoke(
        {
            "prompt": (
                "Analyze all uploaded images.\n\n"
                "For each image, describe it in detail.\n"
                "Extract any visible text.\n"
                "If there are multiple images, compare them and explain similarities, differences, or relationships."
            )
        }
    )


# --------------------------------------------------
# Developer Studio
# --------------------------------------------------

DEVELOPER_STATUS = {
    "Explain Code": "Reviewing your code...",
    "Debug Code": "Debugging your code...",
    "Optimize Code": "Optimizing your code...",
    "Generate Code": "Generating code...",
    "Generate Test Cases": "Generating test cases...",
}


@safe_call
def run_developer_studio(
    language,
    task,
    code_input,
    instructions,
    progress=gr.Progress()
):
    status = DEVELOPER_STATUS.get(task, "Working on your code...")
    progress(0.25, desc=status)

    output = run_developer_task(
        language,
        task,
        code_input,
        instructions
    )

    progress(1.0, desc="Done!")

    return (
        "Done.",
        output
    )


# --------------------------------------------------
# Study Buddy
# --------------------------------------------------

STUDY_STATUS = {
    "Explain Topic": "Explaining topic...",
    "Summarize Notes": "Summarizing notes...",
    "Generate Quiz": "Generating quiz...",
    "Generate Flashcards": "Creating flashcards...",
    "Exam Practice": "Preparing exam practice...",
    "Revision Sheet": "Preparing revision sheet...",
}


@safe_call
def run_study_buddy(
    task,
    topic_notes,
    uploaded_files,
    progress=gr.Progress()
):
    status = STUDY_STATUS.get(task, "Processing study material...")
    progress(0.05, desc=status)

    output = run_study_task(
        task,
        topic_notes,
        uploaded_files,
        progress
    )

    return (
        "Done.",
        output
    )

# --------------------------------------------------
# New Chat
# --------------------------------------------------

@safe_call
def new_chat():
    return (
        gr.update(value=[]),                                    # chatbot
        "Reasoning will appear here after your first message.",
        str(uuid.uuid4()),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value="➤ Send", interactive=True),  # send
        gr.update(value="", interactive=True)         # textbox
    )

def fill_prompt(prompt):
    return prompt


# --------------------------------------------------
# UI
# --------------------------------------------------

css = """
#chat-ui{
    max-width:950px;
    margin:0 auto;
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
}

#chat-ui > div {
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
}

#welcome-card{
    text-align:center;
    padding:48px 50px 20px;
    max-width:850px;
    margin:0 auto 20px auto;
}

#welcome-card h1{
    margin-bottom:8px;
}

#welcome-card h3{
    margin:0 0 12px 0;
}

#welcome-card p{
    margin:8px 0;
}

#welcome-card,
#chat-container{
    transition:opacity .15s ease;
}

.example-btn{
    margin:6px !important;
    border:1px solid #818CF8;
    border-radius: 14px !important;
    height:48px;
    font-weight:600;
}

.example-btn + .example-btn{
    margin-left:12px !important;
}

#chat-input{
    margin-top:18px;
    margin-bottom:12px;
}

#chat-box,
#chat-box *{
    border:none !important;
    box-shadow:none !important;
    background:transparent !important;
}

#new-chat-btn{
    border:1px solid #818CF8;
}

#study-output{
    line-height:1.65;
}

#study-output .markdown,
#study-output .prose{
    font-size:16px;
}

#study-output table{
    display:block;
    width:100%;
    overflow-x:auto;
    border-collapse:collapse;
    margin:14px 0;
}

#study-output th,
#study-output td{
    border:1px solid rgba(129, 140, 248, .28);
    padding:8px 10px;
    vertical-align:top;
}

#study-output pre{
    overflow-x:auto;
}

#study-output mjx-container{
    overflow-x:auto;
    overflow-y:hidden;
    max-width:100%;
    padding:4px 0;
}

#study-output mjx-container[display="true"]{
    margin:16px 0 !important;
}


"""

with gr.Blocks(
    title="CORA • Cognitive Omnitask Reasoning Assistant",
    theme=gr.themes.Soft(),
    css=css
) as demo:

    session_id = gr.State(str(uuid.uuid4()))
    uploaded_pdfs = gr.State([])

    gr.Markdown("""
# CORA
""")

    with gr.Tabs():

        # ==================================================
        # TAB 1 — Hybrid Chat
        # ==================================================

        with gr.Tab("💬 Hybrid Chat"):

            with gr.Row():

                with gr.Column(scale=3):

                    chat_ui = gr.Column(elem_id="chat-ui")

                    with chat_ui:

                        # ---------------- Welcome ----------------

                        welcome = gr.Column(elem_id="welcome-card")

                        with welcome:
                            gr.HTML("""
<div>

<h1>🧠 Welcome to CORA</h1>

<h3>Cognitive Omnitask Reasoning Assistant</h3>

<p>
Reason across
<b>documents</b>,
<b>images</b>,
<b>the web</b>,
and
<b>conversation history</b>.
</p>

</div>
""")

                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1):
                                    chip_pdf = gr.Button("📄 Summarize PDF", scale=1, elem_classes="example-btn")
                                with gr.Column(scale=1):
                                    chip_img = gr.Button("🖼 Compare Images", scale=1, elem_classes="example-btn")

                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1):
                                    chip_web = gr.Button("🌐 AI News", scale=1, elem_classes="example-btn")
                                with gr.Column(scale=1):
                                    chip_graph = gr.Button("📊 Explain Graph", scale=1, elem_classes="example-btn")

                        # ---------------- Chat ----------------

                        chat_container = gr.Column(elem_id="chat-container", visible=False)

                        with chat_container:
                            chatbot = gr.Chatbot(
                                value=[],
                                height=560,
                                show_label=False,
                                elem_id="chat-box"
                            )

                        # ---------------- Input ----------------

                        textbox = gr.Textbox(
                            placeholder="Ask CORA anything...",
                            show_label=False,
                            elem_id="chat-input"
                        )

                        with gr.Row():

                            send = gr.Button(
                                value="➤ Send",
                                variant="primary"
                            )

                            new_chat_btn = gr.Button("🆕 New Chat", elem_id="new-chat-btn")

                with gr.Column(scale=1):

                    with gr.Accordion(
                        "🔍 CORA Reasoning Trace",
                        open=False
                    ):

                        trace_box = gr.Textbox(
                            value="Reasoning will appear here after your first message.",
                            lines=20,
                            interactive=False,
                            show_label=False
                        )
            
            chip_pdf.click(
                lambda: "Summarize my uploaded PDF.",
                outputs=textbox
            ).then(
                chat,
                inputs=[textbox, chatbot, session_id],
                outputs=[
                    textbox,
                    chatbot,
                    trace_box,
                    session_id,
                    welcome,
                    chat_container,
                    send,
                    textbox
                ],
                
            )

            chip_img.click(
                lambda: "Compare these uploaded images.",
                outputs=textbox
            ).then(
                chat,
                inputs=[textbox, chatbot, session_id],
                outputs=[
                    textbox,
                    chatbot,
                    trace_box,
                    session_id,
                    welcome,
                    chat_container,
                    send,
                    textbox
                ],
                
            )

            chip_web.click(
                lambda: "What's happening in AI today?",
                outputs=textbox
            ).then(
                chat,
                inputs=[textbox, chatbot, session_id],
                outputs=[
                    textbox,
                    chatbot,
                    trace_box,
                    session_id,
                    welcome,
                    chat_container,
                    send,
                    textbox
                ],
                
            )

            chip_graph.click(
                lambda: "Explain this graph.",
                outputs=textbox,
            ).then(
                chat,
                inputs=[textbox, chatbot, session_id],
                outputs=[
                    textbox,
                    chatbot,
                    trace_box,
                    session_id,
                    welcome,
                    chat_container,
                    send,
                    textbox
                ],
                
            )

            message_state = gr.State("")

            send.click(
                chat,
                inputs=[
                    textbox,
                    chatbot,
                    session_id
                ],
                outputs=[
                    textbox,
                    chatbot,
                    trace_box,
                    session_id,
                    welcome,
                    chat_container,
                    send,
                    textbox
                ],
                
            )

            textbox.submit(
                chat,
                inputs=[
                    textbox,
                    chatbot,
                    session_id
                ],
                outputs=[
                    textbox,
                    chatbot,
                    trace_box,
                    session_id,
                    welcome,
                    chat_container,
                    send,
                    textbox
                ],
                
            )

            new_chat_btn.click(
                new_chat,
                outputs=[
                    chatbot,
                    trace_box,
                    session_id,
                    welcome,
                    chat_container,
                    send,
                    textbox
                ],
            )

        # ==================================================
        # TAB 2 — Documents
        # ==================================================

        with gr.Tab("📄 Document QA"):

            pdf = gr.File(
                label="Upload PDF(s)",
                file_types=[".pdf"],
                file_count="multiple"
            )

            pdf_status = gr.Textbox(
                value="No documents indexed.",
                label="Indexing Status",
                interactive=False
            )

            pdf_list = gr.Markdown(
                "No PDFs uploaded."
            )

            clear_pdf = gr.Button(
                "🗑 Clear PDFs",
                variant="secondary"
            )

            pdf.upload(
                upload_pdf,
                inputs=[
                    pdf,
                    uploaded_pdfs
                ],
                outputs=[
                    pdf_status,
                    uploaded_pdfs,
                    pdf_list
                ]
            )

            clear_pdf.click(
                clear_pdf_list,
                outputs=[
                    pdf,
                    pdf_status,
                    uploaded_pdfs,
                    pdf_list
                ]
            )

        # ==================================================
        # TAB 3 — Image Studio
        # ==================================================

        with gr.Tab("🖼 Image Studio"):

            image = gr.File(
                file_types=["image"],
                file_count="multiple",
                label="Upload Image(s)"
            )

            image_status = gr.Textbox(
                value="No images uploaded",
                label="Status",
                interactive=False
            )

            image_list = gr.Markdown(
                "No images uploaded."
            )

            analysis_box = gr.Textbox(
                lines=18,
                interactive=False,
                label="Vision Analysis"
            )

            analyze_btn = gr.Button(
                "🔍 Analyze Image(s)",
                variant="primary"
            )

            clear_img = gr.Button(
                "🗑 Remove Image",
                variant="secondary"
            )

            image.upload(
                upload_image,
                inputs=image,
                outputs=[
                    image_status,
                    image_list
                ]
            )

            clear_img.click(
                remove_image,
                outputs=[
                    image,
                    image_status,
                    image_list
                ]
            )

            analyze_btn.click(
                analyze_uploaded_images,
                inputs=[],
                outputs=[analysis_box],
            )

        # ==================================================
        # TAB 4 — Developer Studio
        # ==================================================

        with gr.Tab("Developer Studio"):

            gr.Markdown(
                """
## Developer Studio
Understand, debug, optimize, generate, and test code with focused developer workflows.
"""
            )

            with gr.Row():
                dev_language = gr.Dropdown(
                    choices=[
                        "Auto Detect",
                        "Python",
                        "C++",
                        "Java",
                        "JavaScript",
                        "C#",
                        "Go",
                        "Rust",
                    ],
                    value="Auto Detect",
                    label="Language",
                )

                dev_task = gr.Dropdown(
                    choices=[
                        "Explain Code",
                        "Debug Code",
                        "Optimize Code",
                        "Generate Code",
                        "Generate Test Cases",
                    ],
                    value="Explain Code",
                    label="Task",
                )

            dev_input = gr.Textbox(
                label="Code Input or Program Description",
                lines=14,
                placeholder="Paste code here, or describe the program you want to generate..."
            )

            dev_instructions = gr.Textbox(
                label="Optional Additional Instructions or Error Message",
                lines=4,
                placeholder="Add constraints, expected behavior, stack trace, compiler error, or style preferences..."
            )

            dev_run = gr.Button(
                "Run Developer Task",
                variant="primary"
            )

            dev_status = gr.Textbox(
                value="Ready.",
                label="Status",
                interactive=False
            )

            dev_output = gr.Markdown(
                value="",
                label="Output",
                height=520,
                buttons=["copy"],
                container=True,
                padding=True
            )

            dev_run.click(
                run_developer_studio,
                inputs=[
                    dev_language,
                    dev_task,
                    dev_input,
                    dev_instructions
                ],
                outputs=[
                    dev_status,
                    dev_output
                ]
            )

        # ==================================================
        # TAB 5 — Study Buddy
        # ==================================================

        with gr.Tab("Study Buddy"):

            gr.Markdown(
                """
## Study Buddy
Turn topics, notes, PDFs, slides, screenshots, and handwritten material into learning resources.
"""
            )

            study_task = gr.Dropdown(
                choices=[
                    "Explain Topic",
                    "Summarize Notes",
                    "Generate Quiz",
                    "Generate Flashcards",
                    "Exam Practice",
                    "Revision Sheet",
                ],
                value="Explain Topic",
                label="Task",
            )

            study_input = gr.Textbox(
                label="Topic / Question / Notes",
                lines=12,
                placeholder="Enter a topic, ask a study question, or paste notes..."
            )

            study_files = gr.File(
                label="Optional Study Material",
                file_types=[
                    ".pdf",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".bmp",
                    ".gif",
                    ".tif",
                    ".tiff",
                ],
                file_count="multiple"
            )

            study_run = gr.Button(
                "Run Study Task",
                variant="primary"
            )

            study_status = gr.Textbox(
                value="Ready.",
                label="Status",
                interactive=False
            )

            study_output = gr.Markdown(
                value="",
                label="Output",
                height=520,
                latex_delimiters=[
                    {
                        "left": "\\(",
                        "right": "\\)",
                        "display": False,
                    },
                    {
                        "left": "\\[",
                        "right": "\\]",
                        "display": True,
                    },
                    {
                        "left": "$$",
                        "right": "$$",
                        "display": True,
                    },
                ],
                buttons=["copy"],
                container=True,
                padding=True,
                elem_id="study-output"
            )

            study_run.click(
                run_study_buddy,
                inputs=[
                    study_task,
                    study_input,
                    study_files
                ],
                outputs=[
                    study_status,
                    study_output
                ]
            )


    gr.Markdown("""
---
<div align="center">

🧠 <b>CORA</b> • Cognitive Omnitask Reasoning Assistant

Built with LangGraph • Groq • Gradio

</div>
"""
    )



if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
