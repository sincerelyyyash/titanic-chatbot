from app.llm_agent import extract_relevant_data
import pandas as pd

df = pd.read_csv("data/titanic.csv")

def generate_visualization_data(query):
    """Returns structured data for visualization instead of images."""
    extracted_data = extract_relevant_data({"query": query, "intent_response": query})
    
    if "histogram" in query.lower() and "age" in query.lower():
        return {"age_histogram": df["Age"].dropna().tolist()}      
    return extracted_data["data"]

