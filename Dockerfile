# Start with Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy everything from your computer into the container
COPY . /app

# Install Python dependencies from requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Tell Docker which port the API will use
# Port 8000 is the standard for FastAPI
EXPOSE 8000

# When the container starts, run this command
# This starts the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
