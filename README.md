# Blog MAS — Multi-Agent Blog Generation System

How to Run

  # Navigate to the project directory
  cd blog-mas

  # Install dependencies
  uv sync --group dev

  # Run the test suite
  uv run pytest tests/ -v

  # Run the interactive CLI (requires HF_TOKEN env var for real LLM calls)
  export HF_TOKEN="your_huggingface_api_token"
  uv run blog-mas

  To use a different model, set the environment variable or modify DEFAULT_MODEL in src/blog_mas/llm.py.
