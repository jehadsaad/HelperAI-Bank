FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen

# Copy project files
COPY api ./api
COPY src ./src
COPY database ./database
COPY model ./model
COPY fronted ./fronted
COPY data ./data

# Expose application port
EXPOSE 8000

# Start FastAPI application
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]