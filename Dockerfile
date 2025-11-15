# --------- STAGE 1: The 'uv-builder' ---------
FROM python:3.13.5-slim AS uv-builder

# Use pip to install uv
RUN pip install uv


# --------- STAGE 2: Final Image ---------
FROM python:3.13.5-slim

# Copy the 'uv' executable from the uv-builder stage
COPY --from=uv-builder /usr/local/bin/uv /usr/local/bin/uv

# Copy only the requirements files first for caching purposes
COPY requirements.txt .
COPY requirements-torch.txt .

# Install dependencies using UV
RUN uv pip install -r requirements.txt --system
RUN uv pip install -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu130 --system
RUN sudo apt install libxcb-cursor0

# Copy the application code into the container at /app
COPY . .

# Define the command to run the application
CMD ["python", "main.py"]