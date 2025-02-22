import os
import pandas as pd
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

df = pd.read_csv("data/titanic.csv")

intent_prompt = PromptTemplate.from_template(
    "Analyze the user query: '{query}'. "
    "Determine whether they are asking about survival rate, passenger demographics, ticket prices, or other insights."
)
intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

data_prompt = PromptTemplate.from_template(
    "Given the Titanic dataset and user intent '{intent}', provide a detailed and human-friendly answer. "
    "Ensure your response is clear and insightful."
)
data_chain = LLMChain(llm=llm, prompt=data_prompt)

titanic_chain = SimpleSequentialChain(
    chains=[intent_chain, data_chain], verbose=True
)

def ask_titanic_ai(query: str):
    """Runs the sequential chain and gets the response"""
    return titanic_chain.run(query)
