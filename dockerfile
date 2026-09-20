FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn scikit-learn numpy pickle5

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]