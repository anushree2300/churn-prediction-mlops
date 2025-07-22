FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements-docker.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy necessary files
COPY ./app ./app
COPY ./models ./models

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]