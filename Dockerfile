FROM python:3.11-slim

WORKDIR /app

# Install deploy-only requirements first so this layer is cached
# across rebuilds when only your app code changes.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# App code + anything the runtime needs (see .dockerignore for what's
# excluded -- training scripts, train/test CSVs, local venv, etc.)
COPY . .

RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
# Hugging Face Spaces (Docker SDK) expects the app on port 7860 by default.
EXPOSE 7860

CMD ["./start.sh"]
