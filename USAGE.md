# Usage Guide

This guide will walk you through the complete lifecycle of using the T1D Analytics Suite—from downloading datasets to querying them via the web interface.

---

## 🛑 Step 1: Start the Local LLM (Ollama)

The analytics engine uses a local LLM to translate natural language into SQL. Make sure your local Ollama server is running.

1. Open a terminal.
2. Start the Ollama application or service.
3. Ensure the target model is downloaded:
   ```bash
   ollama pull gemma4
   ```

_(Ollama typically runs automatically in the background on port `11434` once started)._

---

## 📥 Step 2: Download & Extract Data

The project comes with a CLI entry point to manage datasets.

1. Activate your Python virtual environment.
2. Run the `t1d-analytics` CLI to fetch the datasets (Requires appropriate access setups if hitting secured dataset URLs):
   ```bash
   t1d-analytics download
   ```
3. Extract the downloaded Zip files:
   ```bash
   python -c "from t1d_analytics.analytics import extract_zips; extract_zips('data/')"
   ```

---

## 🗄️ Step 3: Populate the DuckDB Database

Once the data is extracted into CSV or TXT formats, you need to load them into the high-performance DuckDB analytics database.

Run the following Python snippet in your terminal to ingest all tabular files in the `data/` directory into a database named `t1d.duckdb`:

```bash
python -c "from t1d_analytics.analytics import load_data_to_duckdb; load_data_to_duckdb('data/', 't1d.duckdb')"
```

This script automatically detects file encodings (UTF-8, UTF-16) and delimiters (comma, tab, pipe) and populates the local `.duckdb` file.

---

## ⚡ Step 4: Start the FastAPI Backend

The web interface communicates with the database through a FastAPI application.

Start the backend server using `uvicorn`:

```bash
# Ensure you are in the root directory of the project
uvicorn src.t1d_analytics.api:app --reload --host 0.0.0.0 --port 8000
```

You should see output indicating that the server has started on `http://0.0.0.0:8000`. You can visit `http://localhost:8000/docs` in your browser to view the interactive API documentation.

---

## 🌐 Step 5: Start the Web UI

With the backend running, open a **new terminal window** to start the frontend application.

```bash
cd web
npm run dev
```

This will launch a Vite development server. Check the terminal output for the local URL (typically `http://localhost:5173/`) and open it in your web browser.

---

## 💬 Step 6: Using the Web Application

### The Interface

- **Sidebar**: Manage your chat sessions here. Click `+ New Chat` to create isolated environments. Click the `⋮` (three dots) next to any chat to **Rename**, **Duplicate**, or **Delete** it.
- **Main Pane**: This is where you converse with the datasets.
- **Model Selector** (Top Right):
  - `Gemma 4`: Type natural language (e.g., _"Show me the first 5 patients in the demographic table"_). The LLM will write the SQL and execute it.
  - `Literal SQL`: Type raw DuckDB SQL directly (e.g., `SELECT * FROM t_demographics LIMIT 5;`).

### Viewing Results

Whenever an executed query returns data, the UI will automatically parse the resulting JSON object and render an interactive, horizontally-scrollable HTML table right inside the chat window. If the query returns empty, it will display a clean _"No results returned"_ message.

---

## 🛠️ Alternate: CLI Interactive REPL

If you prefer staying completely in the terminal, you can bypass the web interface and use the built-in Interactive REPL:

```bash
python -c "from t1d_analytics.analytics import run_query_repl; run_query_repl('t1d.duckdb')"
```

This opens a `query>` prompt.

- If you type a raw SQL command (starting with `SELECT`, `SHOW`, `DESCRIBE`, etc.), it executes it directly.
- If you type a standard sentence, it routes it through the LLM just like the web interface.
- Type `exit` or `quit` to leave.
