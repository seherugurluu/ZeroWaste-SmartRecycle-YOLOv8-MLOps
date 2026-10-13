FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The validation container only needs pytest and the standard-library validator.
RUN pip install --no-cache-dir "pytest>=8.0,<9.0"

COPY src/ src/
COPY tests/ tests/

CMD ["python", "src/validate_dataset.py"]
