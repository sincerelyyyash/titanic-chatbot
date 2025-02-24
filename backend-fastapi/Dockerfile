FROM python:3.9

WORKDIR /app

COPY ./app /app
COPY ./data /app/data  # Ensure the data folder is copied

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
