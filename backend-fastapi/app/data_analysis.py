from app.llm_agent import extract_relevant_data, detect_visualization_type

def generate_visualization_data(query):
    """Generates structured data for visualization based on user query."""
    
    extracted_data = extract_relevant_data({"query": query, "intent_response": query})
    visualization_type = detect_visualization_type(query)

    if extracted_data.get("visualization"):
        extracted_data["visualization"]["type"] = visualization_type or extracted_data["visualization"]["type"]

    return extracted_data

