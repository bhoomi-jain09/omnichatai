import streamlit as st
from l_backend_db import workflow, retrieve_all_threads
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import uuid
import re

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

# =========================
# Utility functions
# =========================

def reset_chat():
    st.session_state["thread_id"] = None
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].append(thread_id)


def load_conversation(thread_id):
    state = workflow.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("message", [])


def generate_thread_title_llm(user_input: str) -> str:
    prompt = f"""
    Generate a short descriptive chat title (max 5 words).
    Use lowercase.
    No punctuation.
    No quotes.

    User message:
    {user_input}
    """
    response = model.invoke(prompt)
    return response.content.strip()


def make_thread_id_from_title(title: str) -> str:
    clean = re.sub(r"[^a-z0-9\s-]", "", title.lower())
    clean = re.sub(r"\s+", "-", clean).strip("-")
    suffix = uuid.uuid4().hex[:4]
    return f"{clean}-{suffix}"


# =========================
# Session state setup
# =========================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None

if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}


# =========================
# Sidebar UI
# =========================

st.sidebar.title("OmniChatAI")
st.sidebar.header("My conversations")

if st.sidebar.button("New chat"):
    reset_chat()

for thread_id in st.session_state["chat_thread"]:

    title = st.session_state["thread_titles"].get(
        thread_id,
        f"chat {str(thread_id)[-4:]}"
    )

    if st.sidebar.button(
        title,
        key=f"thread_btn_{thread_id}"
    ):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        st.session_state["message_history"] = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            }
            for m in messages
        ]


# =========================
# Main chat UI
# =========================

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message…")


# =========================
# Handle user input
# =========================

if user_input:

    # Display user message immediately
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.text(user_input)

    # 🔑 Create thread ONLY on first message
    if st.session_state["thread_id"] is None:
        title = generate_thread_title_llm(user_input)
        thread_id = make_thread_id_from_title(title)

        st.session_state["thread_id"] = thread_id
        st.session_state["thread_titles"][thread_id] = title
        add_thread(thread_id)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    # Stream assistant response
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            chunk.content
            for chunk, _ in workflow.stream(
                {"message": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
