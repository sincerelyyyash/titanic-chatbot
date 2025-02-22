import streamlit as st
import requests
import base64

API_URL = "http://127.0.0.1:8000/chat/"

st.set_page_config(page_title="Titanic Chatbot", page_icon="??", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Titanic Chatbot ")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
        elif message["type"] == "image":
            st.image(message["content"])

query = st.chat_input("Ask a question about the Titanic dataset...")

if query:
    st.session_state.messages.append({"role": "user", "content": query, "type": "text"})
    with st.chat_message("user"):
        st.markdown(query)

    response = requests.post(API_URL, json={"query": query}).json()
    
    bot_response = response.get("answer", "Sorry, I couldn't process that.")
    st.session_state.messages.append({"role": "assistant", "content": bot_response, "type": "text"})
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    image_data = response.get("image", None)
    if image_data:
        image_bytes = base64.b64decode(image_data)
        st.session_state.messages.append({"role": "assistant", "content": image_bytes, "type": "image"})
        
        with st.chat_message("assistant"):
            st.image(image_bytes)
