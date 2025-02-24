import pandas as pd
import numpy as np
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GOOGLE_API_KEY

# Load Titanic dataset
df = pd.read_csv("data/titanic.csv")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Prompt Templates
intent_prompt = PromptTemplate.from_template(
    "User Query: '{query}'\n"
    "Identify all relevant aspects of the Titanic dataset needed to answer the query."
)

data_prompt = PromptTemplate.from_template(
    "Extracted relevant data:\n\n"
    "{data}\n\n"
    "Now, based on this data, provide a detailed yet concise answer to: '{query}'."
)

# Detect visualization type
def detect_visualization_type(query):
    """Infer the type of visualization requested by the user."""
    if isinstance(query, dict):
        query = query.get("query", "")  # Ensure we extract the actual string

    if not isinstance(query, str):
        return None  # If it's still not a string, return None

    query_lower = query.lower()

    if "pie" in query_lower:
        return "pie"
    elif "histogram" in query_lower:
        return "histogram"
    elif "bar" in query_lower:
        return "bar"
    elif "line" in query_lower:
        return "line"
    else:
        return None


def extract_relevant_data(inputs):
    """Extracts relevant data and structures response dynamically based on user query."""
    intent_response = inputs["intent_response"]
    query = inputs["query"]

    intent_text = intent_response.content if hasattr(intent_response, "content") else str(intent_response)
    query_lower = intent_text.lower()

    extracted_data = {}
    visualization = None
    answer = "I couldn't find relevant data for your query."

    # Dynamically detect visualization type
    vis_type = detect_visualization_type(query)  # Only set if user asks for visualization

    if "survival" in query_lower or "survived" in query_lower:
        survival_rate = df.groupby("Pclass")["Survived"].mean().mul(100).round(2).to_dict()
        extracted_data["survival_rate"] = survival_rate
        answer = f"Survival rates by class: {survival_rate}"

        if vis_type:  # Only generate visualization if requested
            visualization = {
                "type": vis_type if vis_type in ["bar", "pie"] else "bar",
                "categories": list(survival_rate.keys()),
                "values": list(survival_rate.values()),
                "title": "Survival Rate by Class",
                "xlabel": "Passenger Class",
                "ylabel": "Survival Rate (%)"
            }

    if "age" in query_lower:
        extracted_data["average_age"] = f"{df['Age'].mean():.1f}"
        extracted_data["age_distribution"] = df['Age'].describe().to_dict()
        answer = f"The average passenger age is {df['Age'].mean():.1f} years."

        # Only create visualization if explicitly requested
        if vis_type:
            hist, bins = np.histogram(df['Age'].dropna(), bins=10)
            visualization = {
                "type": vis_type if vis_type in ["histogram", "bar"] else "histogram",
                "x": bins.tolist(),
                "y": hist.tolist(),
                "title": "Age Distribution",
                "xlabel": "Age",
                "ylabel": "Frequency"
            }

    if "gender" in query_lower:
        gender_distribution = df['Sex'].value_counts(normalize=True).mul(100).round(2).to_dict()
        extracted_data["gender_distribution"] = gender_distribution
        answer = f"Gender distribution: {gender_distribution}"

        if vis_type:
            visualization = {
                "type": vis_type,  
                "categories": list(gender_distribution.keys()),
                "values": list(gender_distribution.values()),
                "title": "Gender Distribution",
                "xlabel": "Gender",
                "ylabel": "Percentage"
            }

    if "class" in query_lower:
        class_distribution = df['Pclass'].value_counts(normalize=True).mul(100).round(2).to_dict()
        extracted_data["class_distribution"] = class_distribution
        answer = f"Passenger class distribution: {class_distribution}"

        if vis_type:
            visualization = {
                "type": vis_type,
                "categories": list(class_distribution.keys()),
                "values": list(class_distribution.values()),
                "title": "Passenger Class Distribution",
                "xlabel": "Class",
                "ylabel": "Percentage"
            }

    if "fare" in query_lower:
        extracted_data["average_fare"] = f"${df['Fare'].mean():.2f}"
        extracted_data["fare_distribution"] = df['Fare'].describe().to_dict()
        answer = f"The average fare was ${df['Fare'].mean():.2f}."

        if vis_type:
            hist, bins = np.histogram(df['Fare'].dropna(), bins=10)
            visualization = {
                "type": vis_type if vis_type in ["histogram", "bar"] else "histogram",
                "x": bins.tolist(),
                "y": hist.tolist(),
                "title": "Fare Distribution",
                "xlabel": "Fare",
                "ylabel": "Frequency"
            }

    return {"answer": answer, "visualization": visualization}

# Create Chain
def create_titanic_chain():
    return RunnableSequence(
        {"query": RunnablePassthrough()}
        | {"intent_response": intent_prompt | llm, "query": RunnablePassthrough()}
        | RunnableLambda(extract_relevant_data)
    )

# Main function
def ask_titanic_ai(query: str):
    chain = create_titanic_chain()
    response = chain.invoke({"query": query})

    if isinstance(response, dict):
        return response  

    return {"answer": str(response), "visualization": None}

