FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull model
RUN ollama serve & sleep 10 && ollama pull llama3

EXPOSE 8501

CMD ollama serve & streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
