import gradio as gr
import requests
import json

API_URL = "http://localhost:8000/api"

# --- Session Helper Functions ---

def fetch_sessions():
    """Fetches active sessions from the API and returns dropdown choices."""
    try:
        response = requests.get(f"{API_URL}/sessions")
        response.raise_for_status()
        session_ids = response.json().get("sessions", [])
    except Exception:
        return gr.update(choices=[], value=None)

    if not session_ids:
        return gr.update(choices=[], value=None)

    choices = []
    for sid in session_ids:
        try:
            meta_resp = requests.get(f"{API_URL}/sessions/{sid}")
            meta_resp.raise_for_status()
            meta = meta_resp.json()
            created = meta.get("created_at", "")[:16].replace("T", " ")
            turns = meta.get("turn_count", 0)
            label = f"{sid[:8]}... | Created: {created} | Turns: {turns}"
            choices.append((label, sid))
        except Exception:
            choices.append((sid[:8] + "...", sid))

    return gr.update(choices=choices, value=None)


def resume_session(selected_session_id):
    """Resumes an existing session: sets state and loads chat history."""
    if not selected_session_id:
        return None, [], "No session selected."

    # Fetch full session data to get messages
    try:
        resp = requests.get(f"{API_URL}/sessions/{selected_session_id}")
        resp.raise_for_status()
        meta = resp.json()
    except Exception as e:
        return None, [], f"Error loading session: {e}"

    # Load the full message history from the session
    try:
        hist_resp = requests.get(f"{API_URL}/sessions/{selected_session_id}/history")
        hist_resp.raise_for_status()
        messages = hist_resp.json().get("messages", [])
    except Exception:
        messages = []

    # Convert backend messages (role: human/ai) to Gradio format (role: user/assistant)
    chat_history = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "human":
            chat_history.append({"role": "user", "content": content})
        elif role == "ai":
            chat_history.append({"role": "assistant", "content": content})

    created = meta.get("created_at", "")[:16].replace("T", " ")
    turns = meta.get("turn_count", 0)
    status_msg = f"**Session Resumed!**\n- Session: `{selected_session_id[:8]}...`\n- Created: {created}\n- Previous turns: {turns}"

    return selected_session_id, chat_history, status_msg


# --- Existing Functions ---

def upload_file(fileobj):
    """Handles file upload via API."""
    if fileobj is None:
        return "Please upload a file first.", gr.update(), gr.update(interactive=False)
        
    try:
        files = {"file": open(fileobj, "rb")}
        response = requests.post(f"{API_URL}/upload", files=files)
        response.raise_for_status()
        data = response.json()
        
        # Save session_id in state
        session_id = data.get("session_id")
        
        msg = f"""
        **Upload Successful!**
        - Filename: {data.get('filename')}
        - Pages: {data.get('num_pages')}
        - Chunks: {data.get('num_chunks')}
        - Processing Time: {data.get('elapsed_time')}s
        """
        return msg, session_id, gr.update(interactive=True)
    except Exception as e:
        return f"Error: {str(e)}", None, gr.update(interactive=False)

def summarize_doc(session_id):
    """Calls the summarize endpoint."""
    if not session_id:
        return "Session not active."
        
    try:
        response = requests.post(f"{API_URL}/summarize", json={"session_id": session_id})
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "No answer returned.")
    except Exception as e:
        return f"Error: {str(e)}"

def chat_stream(message, history, session_id):
    """Chats with SSE streaming yielding to Gradio."""
    if not session_id:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please upload a document or resume a session first."}
        ]
        return
        
    # Append human message
    history = history + [{"role": "user", "content": message}]
    # Append empty assistant message to stream into
    history = history + [{"role": "assistant", "content": ""}]
    
    # Set up streaming request
    try:
        response = requests.post(
            f"{API_URL}/chat/stream", 
            json={"session_id": session_id, "question": message},
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                history[-1]["content"] += decoded_line
                yield history
    except Exception as e:
         history[-1]["content"] = f"Error: {str(e)}"
         yield history

def clear_chat(session_id):
    """Clears history on the server and UI."""
    if session_id:
        try:
             requests.post(f"{API_URL}/history/clear", json={"session_id": session_id})
        except:
             pass
    return []

# Custom CSS for the warning badge
css = """
.warning-badge {
    color: #856404;
    background-color: #fff3cd;
    border-color: #ffeeba;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.875em;
    font-weight: bold;
    display: inline-block;
    margin-top: 5px;
}
"""

with gr.Blocks(title="Intelligent Document Assistant") as demo:
    gr.Markdown("# Intelligent Document Assistant")
    
    session_state = gr.State(None)
    
    with gr.Tabs():
        # Tab 1: Sessions (Resume existing)
        with gr.Tab("Sessions"):
            gr.Markdown("Resume a previous session to continue chatting with an already-uploaded document.")
            
            session_dropdown = gr.Dropdown(
                label="Available Sessions",
                choices=[],
                interactive=True
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Sessions")
                resume_btn = gr.Button("▶ Resume Session", variant="primary")
            
            session_status = gr.Markdown()
            
            # We need a chatbot reference for resume — it's defined in Chat tab below
            # So we wire it up after the Chat tab is created
            
            refresh_btn.click(
                fn=fetch_sessions,
                inputs=[],
                outputs=[session_dropdown]
            )
        
        # Tab 2: Upload
        with gr.Tab("Upload"):
            gr.Markdown("Upload a PDF document to begin a **new** session.")
            file_input = gr.File(label="Select PDF file", file_types=[".pdf"])
            upload_btn = gr.Button("Upload & Process", variant="primary")
            upload_status = gr.Markdown()
            
            gr.Markdown("---")
            summarize_btn = gr.Button("Summarize Document")
            with gr.Accordion("Document Summary", open=False):
                summary_output = gr.Markdown()
            
            upload_btn.click(
                fn=upload_file,
                inputs=[file_input],
                outputs=[upload_status, session_state, summarize_btn]
            )
            
            summarize_btn.click(
                fn=summarize_doc,
                inputs=[session_state],
                outputs=[summary_output]
            )
            
        # Tab 3: Chat
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=500, label="Q&A")
            msg_input = gr.Textbox(placeholder="Ask a question about your document...", label="Your Question")
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear History")
            
            # Streaming UI update
            send_btn.click(
                fn=chat_stream,
                inputs=[msg_input, chatbot, session_state],
                outputs=[chatbot]
            ).then(
                fn=lambda: "", 
                inputs=[], 
                outputs=[msg_input]
            )
            
            # Using submit on textbox as well
            msg_input.submit(
                fn=chat_stream,
                inputs=[msg_input, chatbot, session_state],
                outputs=[chatbot]
            ).then(
                fn=lambda: "", 
                inputs=[], 
                outputs=[msg_input]
            )
            
            clear_btn.click(
                fn=clear_chat,
                inputs=[session_state],
                outputs=[chatbot]
            )
    
    # Wire up the resume button now that chatbot exists
    resume_btn.click(
        fn=resume_session,
        inputs=[session_dropdown],
        outputs=[session_state, chatbot, session_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=css)
