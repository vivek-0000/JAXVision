# One container serves the API and the dashboard.
#
#   docker build -t jaxvision .
#   docker run -p 8000:8000 jaxvision        ->  http://localhost:8000
#
# Requires a trained model in ./artifacts (run `python -m jaxvision.train` first).

# ---- Stage 1: build the React app -------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python server (CPU JAX is enough for inference) ---------------
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY jaxvision/ ./jaxvision/
COPY server/ ./server/
COPY artifacts/ ./artifacts/
COPY --from=web /web/dist ./frontend/dist

# Hosts like Render, Railway and Hugging Face Spaces tell the app which port to use via $PORT.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
