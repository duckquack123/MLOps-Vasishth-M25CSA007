FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirementsQ1.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirementsQ1.txt

# Copy project files
COPY translate.py .
COPY input.rtf .
COPY reference.rtf .

# Run script
CMD ["python", "translate.py"]