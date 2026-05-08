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


  Step-by-step: Running the app
                                                                                                                                     
  Step 1 — Set your HF token in the terminal
                                                                                                                                     
  PowerShell:     
  $env:HF_TOKEN = "hf_your_actual_token_here"

  This works because llm.py and embedding.py both read os.environ.get("HF_TOKEN"). No .env file needed.

  ▎ Important: The default model Qwen/Qwen2.5-7B-Instruct may not be available on HF's free serverless tier. If you get errors,
  ▎ switch to a free-tier model by setting it in src/blog_mas/llm.py line 13, e.g. DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta".

  Step 2 — Start Qdrant via Docker

  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

  Leave this running in a separate terminal. The app connects to http://localhost:6333 by default.

  Step 3 — Install dependencies
  pip install uv  
  uv sync --group dev

  Step 4 — Ingest knowledge files into Qdrant

  This embeds the markdown files from data/knowledge/ (there are 5: AI, climate change, Mediterranean diet, mental health, space
  exploration) into the vector store. This uses LangGraph-style async — the ingestion graph runs chunking, proposition extraction,
  contextualization, embedding, and upsert as a pipeline.

  uv run blog-mas ingest --path data/knowledge

  Step 5 — Ingest blueprints

  This loads the blog-style blueprints (6 JSON files in data/blueprints/) into a separate Qdrant collection.

  uv run blog-mas ingest-blueprints

  Step 6 — Run the interactive CLI

  uv run blog-mas

  You'll see a welcome screen listing available topics. Type a blog request like:

  Blog request > Write a blog about artificial intelligence in healthcare

  The pipeline runs: intake → librarian (RAG search) ∥ researcher (LLM) → writer → validator — with up to 3 revision loops if the
  validator rejects the draft.

  ---
  If you want to redo ingestion from scratch

  Add --rebuild to drop and recreate the Qdrant collections:

  uv run blog-mas ingest --rebuild
  uv run blog-mas ingest-blueprints --rebuild

