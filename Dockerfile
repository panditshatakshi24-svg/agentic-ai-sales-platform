FROM python:3.11

WORKDIR /app

COPY . .

# Install system packages
RUN apt-get update && apt-get install -y \
    curl \
    zstd

# Upgrade pip
RUN pip install --upgrade pip

# Install Python requirements
RUN pip install -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull lightweight model
RUN ollama serve & sleep 15 && ollama pull phi3

EXPOSE 8501

CMD ollama serve & streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
