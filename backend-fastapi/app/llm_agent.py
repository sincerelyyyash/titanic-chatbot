import pandas as pd
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GOOGLE_API_KEY

df = pd.read_csv("data/titanic.csv")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

intent_prompt = PromptTemplate.from_template(
    "User Query: '{query}'\n"
    "Identify all relevant aspects of the Titanic dataset needed to answer the query."
)

data_prompt = PromptTemplate.from_template(
    "Extracted relevant data:\n\n"
    "{data}\n\n"
    "Now, based on this data, provide a detailed yet concise answer to: '{query}'."
)

def extract_relevant_data(inputs):
    intent_response = inputs["intent_response"]
    query = inputs["query"]

    intent_text = intent_response.content if hasattr(intent_response, "content") else str(intent_response)
    query_lower = intent_text.lower()

    extracted_data = {}

    if "survival" in query_lower or "survived" in query_lower:
        extracted_data["survival_rate"] = df.groupby("Pclass")["Survived"].mean().mul(100).round(2).to_dict()

    if "age" in query_lower:
        extracted_data["average_age"] = f"{df['Age'].mean():.1f}"
        extracted_data["age_distribution"] = df['Age'].describe().to_dict()

    if "gender" in query_lower:
        extracted_data["gender_distribution"] = df['Sex'].value_counts(normalize=True).mul(100).round(2).to_dict()

    if "class" in query_lower:
        extracted_data["class_distribution"] = df['Pclass'].value_counts(normalize=True).mul(100).round(2).to_dict()

    if "fare" in query_lower:
        extracted_data["average_fare"] = f"${df['Fare'].mean():.2f}"
        extracted_data["fare_distribution"] = df['Fare'].describe().to_dict()

    if not extracted_data:
        extracted_data["message"] = "Query not recognized. Please refine your question."

    return {"data": extracted_data, "query": query}

def create_titanic_chain():
    return RunnableSequence(
        {"query": RunnablePassthrough()}
        | {"intent_response": intent_prompt | llm, "query": RunnablePassthrough()}
        | RunnableLambda(extract_relevant_data)
        | data_prompt
        | llm
    )

def ask_titanic_ai(query: str):
    chain = create_titanic_chain()
    response = chain.invoke({"query": query})

    if hasattr(response, "content"):
        return response.content
    return str(response)

