# Titanic Chatbot

A chatbot powered by LangChain and Google Gemini AI to analyze and visualize Titanic dataset queries dynamically.

## Features
- Understands and processes user queries related to the Titanic dataset.
- Dynamically identifies required data aspects and determines visualization needs.
- Supports histograms, bar charts, and other visualizations without hardcoding specific data fields.
- Uses **FastAPI** for backend, **Streamlit** for frontend, and **Google Gemini AI** for intelligent query processing.
- Can be run using **Docker Compose**.

## Setup & Installation
### Prerequisites
- **Docker** and **Docker Compose** installed.
- **Google API Key** for using Google Gemini AI.

### Environment Variables
Create a `.env` file in the project root and add:
```
GOOGLE_API_KEY=your_google_api_key
```

## Running the Project
### Using Docker Compose (Recommended)
```sh
docker-compose up --build
```

## Running Manually (Without Docker)
### Backend (FastAPI)
1. Navigate to the backend directory:
```sh
cd backend-flaskapi
```
2. Install dependencies:
```sh
pip install -r requirements.txt
```
3. Run the FastAPI server:
```sh
uvicorn main:app --reload
```

### Frontend (Streamlit)
1. Open a new terminal and navigate to the frontend directory:
```sh
cd frontend-streamlit
```
2. Install dependencies:
```sh
pip install -r requirements.txt
```
3. Run the Streamlit app:
```sh
streamlit run app.py
```

## Technologies Used
- **FastAPI** (Backend API)
- **Streamlit** (Frontend UI)
- **LangChain** (LLM-powered query processing)
- **Google Gemini AI** (Natural Language Processing)
- **Pandas & NumPy** (Data analysis)
- **Matplotlib** (Data visualization)
- **Docker & Docker Compose** (Containerization)
- - **Titanic Dataset** (from kaggle)

## License
MIT License

