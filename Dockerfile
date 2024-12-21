FROM python:3.10
EXPOSE 8501
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "main.py", \
"--server.address=0.0.0.0", \
"--server.headless=true", \
"--server.enableCORS=false", \
"--browser.gatherUsageStats=false"]