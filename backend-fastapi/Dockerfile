FROM python:3.9

WORKDIR /app

COPY ./app /app/app
COPY ./data /app/data  


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
