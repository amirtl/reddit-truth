FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install uv --quiet

# Install dependencies first (layer cache — only re-runs if requirements.txt changes)
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8000
