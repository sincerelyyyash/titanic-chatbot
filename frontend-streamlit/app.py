import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

API_URL = "https://tt.api.sincerelyyyash.com/chat/"

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
    
    visualization_data = response.get("visualization_data")
    if visualization_data is not None:
        visualization_data = visualization_data.get("visualization", None)  # ? Safe lookup

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    if visualization_data:
        st.subheader(visualization_data.get("title", "Data Visualization"))

        vis_type = visualization_data["type"]

        if vis_type == "histogram":
            plt.figure(figsize=(6, 3))
            plt.bar(visualization_data["x"][:-1], visualization_data["y"], width=np.diff(visualization_data["x"]))
            plt.xlabel(visualization_data.get("xlabel", ""))
            plt.ylabel(visualization_data.get("ylabel", ""))
            plt.title(visualization_data.get("title", ""))
            st.pyplot(plt)

        elif vis_type == "bar":
            df = pd.DataFrame({"Category": visualization_data["categories"], "Value": visualization_data["values"]})
            st.bar_chart(df.set_index("Category"), use_container_width=False)

        elif vis_type == "pie":
            plt.figure(figsize=(3, 3))
            plt.pie(visualization_data["values"], labels=visualization_data["categories"], autopct="%1.1f%%", startangle=90)
            plt.title(visualization_data.get("title", ""))
            st.pyplot(plt)

