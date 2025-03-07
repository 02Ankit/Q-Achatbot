__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from rag_function import rag_func

# Set initial message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, how can I help you?"}
    ]

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_prompt = st.chat_input()

if user_prompt:  # Ensuring input is not empty
    # Append user message
    st.session_state.messages.append({
        "role": "user", 
        "content": user_prompt
    })
    with st.chat_message("user"):
        st.write(user_prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = rag_func(user_prompt) or "I'm sorry, I couldn't process your request."  # Handle errors
            st.write(ai_response)

    # Append AI response to session
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)
