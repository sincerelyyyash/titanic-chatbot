import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/chat/"

st.set_page_config(page_title="Titanic Chatbot", page_icon="??", layout="wide")
st.title("Titanic Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about the Titanic dataset...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    response = requests.post(API_URL, json={"query": query}).json()
    bot_response = response.get("answer", "Sorry, I couldn't process that.")
    visualization_data = response.get("visualization_data", None)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    if visualization_data:
        st.subheader("Data Visualization")

        for key, value in visualization_data.items():
            st.markdown(f"### {key.replace('_', ' ').title()}")
            if isinstance(value, dict):
                df = pd.DataFrame.from_dict(value, orient="index", columns=["Value"])
                st.bar_chart(df)
            elif isinstance(value, list):
                st.line_chart(pd.Series(value))
            else:
                st.write(value)

