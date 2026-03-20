FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY gam_service_v2.py ./gam_service.py

# Run
CMD ["python", "gam_service.py"]
