FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first — must happen before requirements.txt
# to prevent pip from pulling the 4GB CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the full project (backend code + models/saved + data)
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "cd backend && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
