import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id, "Untitled")
    st.session_state['message_history'] = []

def add_thread(thread_id, name="Untitled"):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        st.session_state['chat_names'][thread_id] = name

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# **************************************** Session Setup ******************************
if 'rag_mode' not in st.session_state:
    st.session_state['rag_mode'] = False

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'chat_names' not in st.session_state:
    st.session_state['chat_names'] = {}

add_thread(st.session_state['thread_id'])  # ensure current thread exists

# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

if st.sidebar.button('RAG Workflow'):
    st.session_state['rag_mode'] = True
    reset_chat()  # optional: start a fresh conversation for RAG



st.sidebar.header('Chat history')

for i, thread_id in enumerate(st.session_state['chat_threads'][::-1]):
    name = st.session_state['chat_names'].get(thread_id, "Untitled")
    if st.sidebar.button(name, key=f"thread_btn_{thread_id}_{i}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

# show conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # save user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    # auto-generate chat name if still "Untitled"
    if st.session_state['chat_names'][st.session_state['thread_id']] == "Untitled":
        short_name = user_input.strip().split('?')[0][:40]
        st.session_state['chat_names'][st.session_state['thread_id']] = short_name + ("..." if len(user_input) > 40 else "")

    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        if st.session_state.get('rag_mode', False):
            # Use the full RAG pipeline
            rag_pipeline = st.session_state['rag_pipeline']
            ai_message = rag_pipeline.run(user_input)  # or whatever method your pipeline exposes
            st.text(ai_message)
        else:
            # Normal chatbot
            CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
            ai_message = st.write_stream(
                message_chunk.content for message_chunk, metadata in chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode='messages'
                )
            )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
