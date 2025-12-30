"""
Streamlit Frontend for Data Analyst Chatbot
Interactive web interface for analyzing uploaded CSV data.
"""

import os
import sys
import asyncio
import pandas as pd
import streamlit as st
import queue
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to Python path so we can import the package
# __file__ is at: data_analyst_chatbot/data_analyst_chatbot/app.py
# We need: data_analyst_chatbot/ (two levels up)
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables from project root
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

from data_analyst_chatbot.workflow import DataAnalystWorkflow, _runner
from workflows.events import Event
import hashlib

# Page configuration
st.set_page_config(
    page_title="Data Analyst Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional dashboard look
st.markdown("""
    <style>
    /* Global Reset & Typography */
    .stApp {
        background-color: #ffffff;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
        color: #0f172a;
    }

    /* Remove Streamlit's huge top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 80rem !important;
    }
    
    /* Style Header to allow sidebar toggle but keep it clean */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Top Bar Styling */
    .top-bar-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .top-bar-subtitle {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 400;
        background-color: #f1f5f9;
        padding: 0.125rem 0.5rem;
        border-radius: 9999px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* Modern Navigation List Styling */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
    }
    
    .sidebar-logo {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin-bottom: 2.5rem;
    }

    .sidebar-section-header {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* Session Navigation Items */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        border: none !important;
        background-color: transparent !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 0.75rem !important;
        color: #475569 !important;
        border-radius: 0.375rem !important;
        height: auto !important;
        line-height: 1.4 !important;
        transition: all 0.2s ease;
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }

    section[data-testid="stSidebar"] .stButton > button div,
    section[data-testid="stSidebar"] .stButton > button span,
    section[data-testid="stSidebar"] .stButton > button p {
        text-align: left !important;
        margin: 0 !important;
        width: 100% !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }

    /* Active Session Item */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border-left: 2px solid #0f172a !important;
        border-radius: 0 0.375rem 0.375rem 0 !important;
    }

    /* Plus Button / Uploader Integration */
    .upload-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 15vh;
        margin-bottom: 2rem;
    }
    
    .upload-title {
        font-size: 2rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 2rem;
        letter-spacing: -0.025em;
    }

    /* Hide the default uploader text/box logic for the 'plus' style */
    .plus-uploader-container {
        position: fixed;
        bottom: 1.5rem;
        left: calc(50% - 25rem); /* Assuming 80rem max width and sidebar width */
        z-index: 100;
        width: 3rem;
    }
    
    [data-testid="stFileUploader"] {
        padding-top: 0;
    }
    
    [data-testid="stFileUploadDropzone"] {
        border: 1px solid #e2e8f0 !important;
        background-color: white !important;
        border-radius: 2rem !important;
        padding: 1rem !important;
    }

    /* New Analysis Button - Specific Style */
    .new-analysis-btn {
        margin-bottom: 0.5rem !important;
    }
    
    section[data-testid="stSidebar"] .stButton:has(button:contains("+ New Analysis")) button {
        background-color: #0f172a !important;
        color: white !important;
        font-weight: 600 !important;
        justify-content: flex-start !important;
        text-align: left !important;
        margin-bottom: 1rem !important;
        padding-left: 1rem !important;
    }
    
    section[data-testid="stSidebar"] .stButton:has(button:contains("+ New Analysis")) button * {
        color: white !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }

    /* File Info Card in Sidebar */
    .sidebar-file-card {
        margin-top: 0.5rem;
        padding: 0.75rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    /* Card/Container Styling */
    .dashboard-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    .plot-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-bottom: 0.25rem;
        display: flex;
        justify-content: center;
    }
    
    .plot-card img {
        max-height: 450px !important;
        width: auto !important;
        object-fit: contain !important;
    }
    
    .insight-header {
        font-weight: 600;
        color: #0f172a;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #f1f5f9;
    }

    /* Chat Interface Styling */
    .stChatMessage {
        padding: 0.25rem 0;
        margin-bottom: 0rem;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .stChatMessage .stMarkdown {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* User Message Container */
    div[data-testid="stChatMessageUser"] > div {
        background-color: #f1f5f9;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border-top-right-radius: 0;
    }
    
    /* Assistant Message Container */
    div[data-testid="stChatMessageAssistant"] > div {
        background-color: transparent;
        padding: 0 1rem 0.25rem 1rem;
    }

    /* Reduce spacing for expanders and tabs in chat */
    .stChatMessage div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stChatMessage .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
    }

    /* Input Area Styling */
    .stChatInputContainer {
        padding-bottom: 1rem;
    }
    
    .stChatInputContainer textarea {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 0.5rem;
        color: #0f172a;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .stChatInputContainer textarea:focus {
        border-color: #0f172a;
        box-shadow: 0 0 0 1px #0f172a;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 0.375rem;
        font-weight: 500;
        font-size: 0.875rem;
        height: 2.25rem;
        border-color: #cbd5e1;
        background-color: white;
        transition: all 0.15s ease-in-out;
    }
    
    .stButton > button:hover {
        background-color: #f8fafc;
        border-color: #94a3b8;
    }

    .stButton > button:hover {
        background-color: #f8fafc;
        border-color: #94a3b8;
    }

    /* Status Container */
    div[data-testid="stStatusWidget"] {
        border: 1px solid #e2e8f0;
        background-color: #fff;
        border-radius: 0.5rem;
    }
    
    /* Hide Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Define Avatar Paths (optional - will use Streamlit defaults if not found)
# Check if avatar files exist, otherwise use None for default avatars
_ai_avatar = "/Users/nathanielislas/.gemini/antigravity/brain/fbe0f66c-cd06-46e2-aff6-cc29ce0c8479/ai_avatar_icon_1767120510418.png"
_user_avatar = "/Users/nathanielislas/.gemini/antigravity/brain/fbe0f66c-cd06-46e2-aff6-cc29ce0c8479/user_avatar_icon_1767120522427.png"
AI_AVATAR_PATH = _ai_avatar if Path(_ai_avatar).exists() else None
USER_AVATAR_PATH = _user_avatar if Path(_user_avatar).exists() else None

@st.cache_resource
def get_workflow():
    """Initialize the workflow."""
    return DataAnalystWorkflow(timeout=600)

async def generate_dataset_description_async(file_path: str) -> Tuple[Dict[str, Any], str]:
    """Generate dataset metadata and description asynchronously."""
    from data_analyst_chatbot.utils.data_loader import get_csv_metadata, generate_dataset_description
    workflow = DataAnalystWorkflow(timeout=600)
    metadata = get_csv_metadata(file_path)
    description = await generate_dataset_description(metadata, workflow.llm)
    return metadata, description

def save_session():
    """Save the current session state to disk."""
    if "session_folder" not in st.session_state or not st.session_state.session_folder:
        return
    
    session_folder = Path(st.session_state.session_folder)
    session_folder.mkdir(exist_ok=True)
    
    # Data to persist
    state_to_save = {
        "file_path": st.session_state.get("file_path"),
        "last_file_hash": st.session_state.get("last_file_hash"),
        "messages": st.session_state.get("messages", []),
        "dataset_metadata": st.session_state.get("dataset_metadata"),
        "dataset_description": st.session_state.get("dataset_description"),
        "last_result": st.session_state.get("last_result"),
        "timestamp": datetime.now().isoformat(),
        "display_name": Path(st.session_state.get("file_path")).name if st.session_state.get("file_path") else "Unknown"
    }
    
    with open(session_folder / "session_state.json", "w") as f:
        json.dump(state_to_save, f, indent=2, default=str)

def list_sessions():
    """List all available saved sessions."""
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        return []
    
    sessions = []
    for s_dir in sessions_dir.iterdir():
        if s_dir.is_dir():
            state_file = s_dir / "session_state.json"
            if state_file.exists():
                try:
                    with open(state_file, "r") as f:
                        data = json.load(f)
                        data["id"] = s_dir.name
                        sessions.append(data)
                except:
                    continue
    
    # Sort by timestamp, newest first
    return sorted(sessions, key=lambda x: x.get("timestamp", ""), reverse=True)

def load_session(session_id: str):
    """Load a specific session into st.session_state."""
    session_path = Path("sessions") / session_id / "session_state.json"
    if not session_path.exists():
        return
    
    with open(session_path, "r") as f:
        data = json.load(f)
    
    st.session_state.messages = data.get("messages", [])
    st.session_state.file_path = data.get("file_path")
    st.session_state.last_file_hash = data.get("last_file_hash")
    st.session_state.dataset_metadata = data.get("dataset_metadata")
    st.session_state.dataset_description = data.get("dataset_description")
    st.session_state.last_result = data.get("last_result")
    st.session_state.session_folder = str(Path("sessions") / session_id)
    st.rerun()

def render_plot_gallery(plot_paths: List[str]):
    """Render a clean gallery of plots using tabs for multiple images."""
    if not plot_paths:
        return
        
    # Filter for valid existing files that are not empty (blank images are typically > 2KB)
    valid_paths = [
        p for p in plot_paths 
        if Path(p).exists() and Path(p).stat().st_size > 2000
    ]
    
    if not valid_paths:
        return

    st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
    
    if len(valid_paths) == 1:
        # Single plot: Show in a compact card
        st.markdown('<div class="plot-card">', unsafe_allow_html=True)
        st.image(valid_paths[0], width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Multiple plots: Wrap in a collapsible expander with tabs
        with st.expander("Analytical Visualizations", expanded=False):
            tab_names = [f"Plot {i+1}" for i in range(len(valid_paths))]
            tabs = st.tabs(tab_names)
            for i, tab in enumerate(tabs):
                with tab:
                    st.markdown('<div class="plot-card" style="border: none;">', unsafe_allow_html=True)
                    st.image(valid_paths[i], width="stretch")
                    st.markdown('</div>', unsafe_allow_html=True)

async def create_and_run_workflow(query: str, file_path: str, event_queue: queue.Queue, cached_metadata: dict = None, cached_description: str = None, chat_history: List[Dict[str, str]] = None, log_file_path: str = None, session_folder: str = None):
    """Create workflow and run analysis in the background thread's event loop."""
    loop = asyncio.get_event_loop()
    assert loop is not None, "Event loop must be set"
    
    workflow = DataAnalystWorkflow(timeout=600, log_file_path=log_file_path, max_iterations=3, session_folder=session_folder)
    
    if cached_metadata is None or cached_description is None:
        cached_metadata, cached_description = await generate_dataset_description_async(file_path)
    
    handler = workflow.run(
        query=query, 
        file_path=file_path, 
        cached_metadata=cached_metadata, 
        cached_description=cached_description,
        chat_history=chat_history or []
    )
    
    async for event in handler.stream_events():
        if hasattr(event, "msg"):
            event_queue.put(event.msg)
            
    result = await handler
    return result

def run_analysis(query: str, file_path: str, chat_history: List[Dict[str, str]] = None):
    """Run analysis using the background thread runner and stream events."""
    event_queue = queue.Queue()
    _runner.start()
    
    import time
    max_wait = 2.0
    wait_interval = 0.01
    waited = 0.0
    while _runner._loop is None and waited < max_wait:
        time.sleep(wait_interval)
        waited += wait_interval
    
    if _runner._loop is None:
        raise RuntimeError("Event loop not initialized.")
    
    if "dataset_cache" not in st.session_state:
        st.session_state.dataset_cache = {}
    
    cached_metadata = None
    cached_description = None
    
    try:
        file_stat = Path(file_path).stat()
        cache_key = f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}"
        cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        if cache_key_hash in st.session_state.dataset_cache:
            cached = st.session_state.dataset_cache[cache_key_hash]
            cached_metadata = cached["metadata"]
            cached_description = cached["description"]
    except (OSError, FileNotFoundError):
        pass
    
    log_file_path = st.session_state.get("log_file_path")
    session_folder = st.session_state.get("session_folder")
    
    future = asyncio.run_coroutine_threadsafe(
        create_and_run_workflow(query, file_path, event_queue, cached_metadata, cached_description, chat_history, log_file_path, session_folder), 
        _runner._loop
    )
    
    with st.status("Analyzing...", expanded=False) as status:
        events = []
        last_msg = ""
        while not future.done():
            try:
                msg = event_queue.get(timeout=0.1)
                if msg and msg != last_msg:
                    events.append(msg)
                    last_msg = msg
                    # Show a much more descriptive label without aggressive truncation
                    status.update(label=f"Analysis: {msg[:160]}..." if len(msg) > 160 else f"Analysis: {msg}", state="running")
                    # Write the full, non-truncated message to the status container
                    st.write(f"• {msg}")
            except queue.Empty:
                continue
        while not event_queue.empty():
            msg = event_queue.get_nowait()
            if msg and msg not in events:
                events.append(msg)
                st.write(f"• {msg}")
        status.update(label="Analysis Complete", state="complete")
        
    result = future.result()
    
    if cached_metadata is None or cached_description is None:
        try:
            file_stat = Path(file_path).stat()
            cache_key = f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}"
            cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            
            if "description" in result and "metadata" in result:
                st.session_state.dataset_cache[cache_key_hash] = {
                    "metadata": result["metadata"],
                    "description": result["description"]
                }
        except (OSError, FileNotFoundError):
            pass
    
    return events, result

@st.cache_data
def load_data(file_path):
    """Cached CSV loading to prevent redundant I/O."""
    return pd.read_csv(file_path)

def main():
    # Sidebar: Minimal and Professional
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">Data Analyst</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section-header">Connection</div>', unsafe_allow_html=True)
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            st.markdown('<div style="font-size: 0.85rem; color: #059669; font-weight: 500; padding-left: 0.75rem;">● Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size: 0.85rem; color: #dc2626; font-weight: 500; padding-left: 0.75rem;">○ Disconnected</div>', unsafe_allow_html=True)
        
        # Recent Sessions Section
        st.markdown('<div class="sidebar-section-header">History</div>', unsafe_allow_html=True)
        
        if st.button("+ New Analysis", width="stretch"):
            # Reset everything to start fresh
            for key in ["messages", "file_path", "last_file_hash", "dataset_metadata", "dataset_description", "last_result", "session_folder"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        sessions = list_sessions()
        if sessions:
            for s in sessions[:5]: # Show last 5
                # Simple name: truncate if too long
                display_name = s['display_name']
                if len(display_name) > 20:
                    display_name = display_name[:17] + "..."
                
                # Highlight active session
                session_full_id = str(Path("sessions") / s["id"])
                is_active = st.session_state.get("session_folder") == session_full_id
                
                if st.button(display_name, key=f"session_{s['id']}", width="stretch", type="primary" if is_active else "secondary"):
                    load_session(s["id"])
        else:
            st.markdown('<div style="font-size: 0.75rem; color: #94a3b8; padding-left: 0.75rem;">No recent history</div>', unsafe_allow_html=True)

        # Restore Source Info to Sidebar when a file is active
        if "file_path" in st.session_state:
            st.markdown('<div class="sidebar-section-header">Source</div>', unsafe_allow_html=True)
            df_preview = load_data(st.session_state.file_path)
            st.markdown(f"""
                <div class="sidebar-file-card">
                    <div style="font-weight: 600; font-size: 0.8rem; color: #0f172a; word-break: break-all;">{Path(st.session_state.file_path).name}</div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.125rem;">
                        {len(df_preview):,} rows
                    </div>
                </div>
            """, unsafe_allow_html=True)
                
            with st.expander("Preview Data", expanded=False):
                st.dataframe(df_preview.head(), width="stretch")

    # Main Content
    if "file_path" in st.session_state:
        # Top Bar Layout
        col_title, col_plus, col_actions = st.columns([1, 0.1, 0.2])
        uploaded_name = Path(st.session_state.file_path).name
        
        with col_title:
             st.markdown(f"""
                <div class="top-bar-title">
                    Analysis
                    <span class="top-bar-subtitle">{uploaded_name}</span>
                </div>
            """, unsafe_allow_html=True)

        with col_plus:
            # Persistent "plus" button for new uploads
            with st.popover("＋", help="Upload a new dataset"):
                new_file = st.file_uploader("Upload CSV", type="csv", key="chat_uploader")
                if new_file:
                    # Trigger the same logic (this should be abstracted, but for now duplicate)
                    file_bytes = new_file.getvalue()
                    file_hash = hashlib.md5(file_bytes).hexdigest()
                    temp_dir = Path("temp_data")
                    temp_dir.mkdir(exist_ok=True)
                    f_path = temp_dir / new_file.name
                    
                    with open(f_path, "wb") as f:
                        f.write(file_bytes)
                    
                    st.session_state.messages = []
                    st.session_state.last_result = None
                    st.session_state.file_path = str(f_path)
                    st.session_state.last_file_hash = file_hash
                    
                    # New session folder
                    sessions_dir = Path("sessions")
                    sessions_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    session_folder = sessions_dir / f"{new_file.name.replace(' ', '_')}_{timestamp}"
                    session_folder.mkdir(exist_ok=True)
                    st.session_state.session_folder = str(session_folder)
                    
                    st.rerun()
            
        with col_actions:
            if st.button("Reset Chat", width="stretch"):
                st.session_state.messages = []
                st.session_state.last_result = None
                st.rerun()

        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        
        # Initial Dataset Context (if no chat history or just loaded)
        if "last_result" in st.session_state and st.session_state.last_result is not None and not st.session_state.get("messages"):
             res = st.session_state.last_result
             with st.expander("Dataset Overview & Insights", expanded=False):
                st.markdown(f"""
                    <div class="dashboard-card">
                        <div style="font-size: 0.9rem; line-height: 1.6; color: #334155;">
                            {res["description"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            avatar_path = USER_AVATAR_PATH if message["role"] == "user" else AI_AVATAR_PATH
            with st.chat_message(message["role"], avatar=avatar_path):
                st.markdown(message["content"])
                
                # Render plots using the new gallery system
                if message["role"] == "assistant" and "plot_paths" in message and message["plot_paths"]:
                    render_plot_gallery(message["plot_paths"])
                
                # Steps
                if message["role"] == "assistant" and "steps" in message and message["steps"]:
                    with st.expander("View Analysis Steps", expanded=False):
                        for step in message["steps"]:
                            st.caption(f"• {step}")

        # Chat Input
        if prompt := st.chat_input("Ask a question about your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=USER_AVATAR_PATH):
                st.markdown(prompt)
            
            with st.chat_message("assistant", avatar=AI_AVATAR_PATH):
                # Placeholder for streaming/processing
                placeholder = st.empty()
                with placeholder.container():
                    st.markdown("<span style='color:#64748b; font-size: 0.9rem;'>Thinking...</span>", unsafe_allow_html=True)
                
                try:
                    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    events, result = run_analysis(prompt, st.session_state.file_path, chat_history)
                    
                    placeholder.empty()
                    
                    response = result["response"]
                    st.markdown(response)
                    
                    plot_paths = result.get("plot_paths")
                    if plot_paths:
                        render_plot_gallery(plot_paths)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "steps": [e for e in events if e],
                        "plot_paths": plot_paths
                    })
                    
                    # Auto-save after assistant response
                    save_session()
                    
                except Exception as e:
                    placeholder.empty()
                    st.error(f"Analysis failed: {e}")

    else:
        # Empty State
        st.markdown(f"""
            <div class="upload-section">
                <div class="upload-title">Analyze. Visualize. Discover.</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Center the uploader
        col_l, col_m, col_r = st.columns([1, 2, 1])
        with col_m:
            uploaded_file = st.file_uploader("Upload a CSV to start analyzing", type="csv")
            
            if uploaded_file:
                # Use content hash to detect real changes and prevent redundant processing
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()
                temp_dir = Path("temp_data")
                temp_dir.mkdir(exist_ok=True)
                file_path = temp_dir / uploaded_file.name
                
                # Detect if this is a actually a new file for this session
                is_new_file = st.session_state.get("last_file_hash") != file_hash
                
                if is_new_file:
                    # Only write to disk if it's new
                    with open(file_path, "wb") as f:
                        f.write(file_bytes)
                    
                    # Reset session state for new file
                    st.session_state.messages = []
                    st.session_state.last_result = None
                    st.session_state.file_path = str(file_path)
                    st.session_state.last_file_hash = file_hash
                    
                    # Generate a single session folder for this specific upload
                    sessions_dir = Path("sessions")
                    sessions_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_filename = uploaded_file.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                    session_folder = sessions_dir / f"{safe_filename}_{timestamp}"
                    session_folder.mkdir(exist_ok=True)
                    st.session_state.session_folder = str(session_folder)
                    
                    # Trigger profiling ONLY on new upload
                    if "dataset_cache" not in st.session_state:
                        st.session_state.dataset_cache = {}
                    
                    cache_key = f"{file_path}_{file_hash}" # Use hash for stable cache key
                    
                    if cache_key in st.session_state.dataset_cache:
                        cached = st.session_state.dataset_cache[cache_key]
                        st.session_state.dataset_metadata = cached["metadata"]
                        st.session_state.dataset_description = cached["description"]
                        st.session_state.last_result = cached
                    else:
                        _runner.start()
                        with st.spinner("Profiling dataset..."):
                            future = asyncio.run_coroutine_threadsafe(
                                generate_dataset_description_async(str(file_path)),
                                _runner._loop
                            )
                            metadata, description = future.result(timeout=60)
                            
                            cached_data = {
                                "metadata": metadata,
                                "description": description,
                                "response": "Ready for analysis."
                            }
                            st.session_state.dataset_cache[cache_key] = cached_data
                            st.session_state.dataset_metadata = metadata
                            st.session_state.dataset_description = description
                            st.session_state.last_result = cached_data
                    
                    # Auto-save after profiling / loading
                    save_session()
                    st.rerun()

if __name__ == "__main__":
    main()
