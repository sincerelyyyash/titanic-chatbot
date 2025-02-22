import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

df = pd.read_csv("data/titanic.csv")

def generate_visual(feature):
    """Generate a histogram for the requested feature."""
    plt.figure(figsize=(6, 4))
    df[feature].hist(bins=20, color='skyblue', edgecolor='black')
    plt.xlabel(feature.capitalize())
    plt.ylabel("Count")
    plt.title(f"Distribution of {feature.capitalize()}")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    
    return encoded_image

