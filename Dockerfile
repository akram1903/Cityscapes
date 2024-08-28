FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

# Set the working directory in the container
WORKDIR /app/

# Copy the requirements (if any additional packages are needed)
# COPY requirements.txt .
COPY  requirements.txt .
# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt


# Copy the res t of the application code to the container
COPY . .


# Expose a port if your model serves an API (e.g., Flask, FastAPI)
# EXPOSE 5000

# Command to run your application

ENTRYPOINT ["python", "app/model.py"]

