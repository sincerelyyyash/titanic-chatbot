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

def prepare_dataset_summary(df):
    """Create a concise summary of the dataset for context"""
    summary = {
        'total_passengers': len(df),
        'survival_rate': f"{(df['Survived'].mean() * 100):.1f}%",
        'avg_age': f"{df['Age'].mean():.1f}",
        'gender_dist': df['Sex'].value_counts().to_dict(),
        'class_dist': df['Pclass'].value_counts().to_dict(),
        'avg_fare': f"{df['Fare'].mean():.2f}"
    }
    return summary

df = pd.read_csv("data/titanic.csv")
dataset_summary = prepare_dataset_summary(df)

intent_prompt = PromptTemplate.from_template(
    "Dataset Context:\n"
    "The Titanic dataset contains information about {total_passengers} passengers.\n"
    "Key metrics: {survival_rate} survival rate, average age {avg_age} years, "
    "average fare ${avg_fare}.\n\n"
    "Analyze the user query: '{query}'. "
    "Determine whether they are asking about survival rate, passenger demographics, "
    "ticket prices, or other insights. Be specific about what aspects of the data "
    "will be needed to answer their question."
)

data_prompt = PromptTemplate.from_template(
    "Dataset Context:\n"
    "- Total passengers: {total_passengers}\n"
    "- Survival rate: {survival_rate}\n"
    "- Average age: {avg_age} years\n"
    "- Gender distribution: {gender_dist}\n"
    "- Class distribution: {class_dist}\n"
    "- Average fare: ${avg_fare}\n\n"
    "Given the above Titanic dataset statistics and user intent '{intent}', "
    "provide a detailed and human-friendly answer. "
    "Use specific numbers from the dataset context to support your response. "
    "If you need to calculate additional statistics, specify what calculations are needed."
)

def create_titanic_chain():
    intent_chain = LLMChain(
        llm=llm,
        prompt=intent_prompt.partial(**dataset_summary)
    )
    
    data_chain = LLMChain(
        llm=llm,
        prompt=data_prompt.partial(**dataset_summary)
    )
    
    return SimpleSequentialChain(
        chains=[intent_chain, data_chain],
        verbose=True
    )

def ask_titanic_ai(query: str):
    """Runs the sequential chain and gets the response"""
    chain = create_titanic_chain()
    return chain.run(query)
