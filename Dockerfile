FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies first for better cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the environment file and application code
COPY .env .env
COPY cosmos_server.py .
COPY cosmos_mcpconfig.py .

# Expose the port
EXPOSE 8050

# Run the app
CMD ["python", "cosmos_server.py"]
 