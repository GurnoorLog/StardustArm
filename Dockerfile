FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglfw3 \
    libegl1-mesa \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["python", "app.py"]
