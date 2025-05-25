# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements directory and install base dependencies
COPY ./requirements/ /app/requirements/
RUN pip install --no-cache-dir -r requirements/base.txt

# Copy the rest of the application code into the container
COPY ./src /app/src
COPY pyproject.toml setup.py README.md /app/

# Install the application
RUN pip install --no-cache-dir .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (optional, can be set in docker-compose.yml or .env)
# ENV NAME World

# Run api_main.py when the container launches
# The entry point is src/rt_rag/api_main.py and the FastAPI app instance is named 'app'
CMD ["uvicorn", "rt_rag.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
