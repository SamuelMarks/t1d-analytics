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

## 📥 Step 2: Download, Extract, or Seed Data

The `t1d-analytics` CLI supports localization via `-l` / `--lang {en,ja,ar,he}` before any subcommand.

1. Activate your Python virtual environment.
2. Run `t1d-analytics download` to fetch datasets from the public repository (supports `--token`, `--user`/`--password`, `--headers`, and `--auth-plugin module:func` for secured endpoints):
   ```bash
   t1d-analytics download -o ./data --concurrency 4
   ```
3. Extract downloaded Zip archives (with optional decompression size guards and tabular-only filtering):
   ```bash
   t1d-analytics extract -d data/ --max-decompressed-gb 200.0 --tabular-only
   ```
4. **(Optional) Seed Baseline Clinical Cohorts**: To immediately populate deterministic baseline T1D tables (`patients`, `cgm_readings`, `hba1c_labs`, `dka_adverse_events`) and canonical views without downloading external archives:
   ```bash
   t1d-analytics seed -d data/ --db t1d_analytics.duckdb --sessions-db data/sessions.duckdb
   ```

---

## 🗄️ Step 3: Populate, Profile, and Export the DuckDB Database

Once data files are extracted into CSV or TXT formats, ingest them into the DuckDB analytics database using `load`. For example, to load data from `/Volumes/TOSHIBA_EXT/stanford/t1d/jaeb` into a `t1d.duckdb` file in the same directory:

```bash
t1d-analytics load --data-dir /Volumes/TOSHIBA_EXT/stanford/t1d/jaeb --db /Volumes/TOSHIBA_EXT/stanford/t1d/jaeb/t1d.duckdb
```

This command automatically detects file encodings (UTF-8, UTF-16) and delimiters (comma, tab, pipe) and populates the local `.duckdb` file.

### Ingesting Proprietary Clinical Formats (SAS & SPSS)

If your clinical trial data includes SAS transport files (`.xpt`), SAS datasets (`.sas7bdat`), or SPSS files (`.sav`), pass the respective inclusion flags:

```bash
t1d-analytics load \
    --data-dir ./data/jaeb_raw \
    --db t1d_analytics.duckdb \
    --include-sas \
    --include-spss \
    --prefix-subdirs
```

You can view the historical audit log of all ingested files and row counts using:

```bash
t1d-analytics manifest --db t1d_analytics.duckdb
```

### Table Profiling, Exporting, & Directory Watching

- **Profile table quality and missingness:**
  ```bash
  t1d-analytics profile --db t1d_analytics.duckdb --table patients
  ```
- **Export any database table to Excel (`.xlsx`), Parquet, or CSV:**
  ```bash
  t1d-analytics export \
      --db t1d_analytics.duckdb \
      --table patients \
      --format excel \
      --output-file ./patients.xlsx
  ```
- **Watch a directory for newly arrived clinical files and auto-ingest:**
  ```bash
  t1d-analytics watch --data-dir ./data/incoming --db t1d_analytics.duckdb --interval 5.0
  ```

---

## 🩺 System Diagnostics (Doctor)

Run the CLI doctor command at any time to verify DuckDB database connectivity, minimum table/row thresholds, and Ollama LLM availability (supports `--json` and `--allow-empty`):

```bash
t1d-analytics doctor --db t1d_analytics.duckdb --min-tables 4 --min-rows 400
```

---

## 🤖 Step 4: Generate Training Data, Train, & Evaluate (Optional)

If you are preparing to fine-tune `gemma-4-sql` on the T1D schema, you can generate synthetic training pairs, export shards, bridge to `gemma-4-sql`, or orchestrate end-to-end training directly from the CLI.

### 1. Generate & Export Training Pairs

```bash
# Generate pretrain_data, sft_data, and dpo_data tables (with stratified _train, _val, _test splits)
t1d-analytics generate-training-data \
    --db t1d_analytics.duckdb \
    --num-pairs 1000 \
    --model gemma4 \
    --provider ollama \
    --split-ratios "0.8,0.1,0.1"

# Export a generated training table to JSONL or Parquet
t1d-analytics export-training-data \
    --db t1d_analytics.duckdb \
    --table sft_data \
    --format parquet \
    --output-file ./training_output/sft_data.parquet
```

### 2. Orchestrate Training & Gemma-4-SQL Bridge

```bash
# End-to-end schema extraction, pair generation, split validation, and training execution
t1d-analytics train \
    --db t1d_analytics.duckdb \
    --output-dir ./training_output \
    --samples-per-table 20 \
    --backend local-cpu \
    --use-lora \
    --split-validation \
    --execute-training

# Or interact directly with the external gemma-4-sql toolchain
t1d-analytics bridge-gemma-sql check
t1d-analytics bridge-gemma-sql etl --stage sft --db t1d_analytics.duckdb --table sft_data
t1d-analytics bridge-gemma-sql train --stage sft --db t1d_analytics.duckdb --table sft_data
```

### 3. Evaluate Model Accuracy

```bash
t1d-analytics evaluate \
    --db t1d_analytics.duckdb \
    --test-set test_benchmarks.json \
    --model gemma4 \
    --output evaluation_report.md
```

### Local Ollama vs. Remote TPU Multi-Node Execution

- **Local Development / Prototyping:**
  Use Ollama (`gemma:2b` or `gemma4`) for local pair generation and query evaluation via `t1d-analytics evaluate`.
- **Distributed TPU Training:**
  Follow [DEPLOY_TO_TPU.md](DEPLOY_TO_TPU.md) and [REPORT.md](REPORT.md), or run `t1d-analytics train --backend remote-tpu-maxtext --tpu-type v4-8 --tpu-zone us-central2-b --gcs-bucket gs://t1d-analytics-artifacts`.

---

## ⚡ Step 5: Start the FastAPI Backend

The web interface communicates with the database through a FastAPI application.

Start the backend server using `uvicorn`, and specify your database path using the `T1D_DB_PATH` environment variable (or set `T1D_DB_SEARCH_DIR` to search `/Volumes/TOSHIBA_EXT/stanford/t1d` locally and `/mnt/data/t1d` on the remote server):

```bash
# Ensure you are in the root directory of the project
T1D_DB_PATH=/Volumes/TOSHIBA_EXT/stanford/t1d/jaeb/t1d.duckdb \
T1D_DB_SEARCH_DIR=/Volumes/TOSHIBA_EXT/stanford/t1d:/mnt/data/t1d \
uvicorn t1d_analytics.api:app --app-dir t1d-analytics-backend/src --reload --host 0.0.0.0 --port 8000
```

You should see output indicating that the server has started on `http://0.0.0.0:8000`. You can visit `http://localhost:8000/docs` in your browser to view the interactive OpenAPI documentation covering `/api/chat`, `/api/chat/stream`, `/api/execute-sql`, `/api/eval/nl-to-sql`, `/api/sessions`, `/api/schema`, `/api/databases`, `/api/table/{name}`, `/api/table/{name}/profile`, `/api/export/{csv,json,excel,parquet}`, `/api/training/jobs`, `/api/models`, `/api/providers/status`, `/api/status`, `/api/ready`, and `/api/health`.

---

## 🌐 Step 6: Start the Web UI

With the backend running, open a **new terminal window** to start the frontend application.

```bash
cd t1d-analytics-web
npm run dev
```

This will launch a Vite development server. Check the terminal output for the local URL (typically `http://localhost:5173/`) and open it in your web browser.

---

## 💬 Step 7: Using the Web Application

### The Interface

- **Sidebar & Deep-Linked Sessions**: Manage persistent chat sessions synchronized with the backend. Click `+ New Chat` to create isolated conversations, or use the `⋮` menu to **Rename**, **Duplicate**, or **Delete** sessions. Active chats automatically sync with the URL hash (`#chat-<id>`) for bookmarking and sharing.
- **Schema Explorer & Database Switcher**: Browse available `.duckdb` files in the workspace, search tables and column types, and click any table to open the **Table Preview & Profiling Modal** (featuring server-side sorting, filtering, data-quality completeness metrics, and one-click CSV, JSON, Excel `.xlsx`, and Parquet downloads).
- **Visual Cohort Builder & Template Library**:
  - Use the **Cohort Builder** modal to construct multi-table `JOIN`, `WHERE`, and aggregation queries visually without writing SQL by hand.
  - Open the **Prompt & SQL Template Picker** (or Table 1 clinical benchmark cards on the empty state) to run curated T1D clinical trial queries and save custom reusable templates.
- **Model Selector & Multi-Provider Settings** (Top Right):
  - Choose between local Ollama models (`gemma:2b`, `llama3.2:1b`, `phi4-mini`, `gemma4`), cloud `any-llm` providers (`HuggingFace`, `OpenAI`, `Anthropic`, `Google`), or `Literal SQL` mode.
  - Click the **Provider Settings (⚙️)** button to configure API keys or inspect live provider connectivity. If an LLM is offline or unreachable, the backend automatically synthesizes schema-guided fallback SQL and displays a diagnostic banner with the underlying LLM error reason.
- **Internationalization (i18n)**: Switch UI language across 8 supported locales (`en`, `es`, `fr`, `de`, `zh`, `ja`, `ar`, `he`), with automatic Right-to-Left (RTL) layout adaptation for Arabic and Hebrew.

### Viewing Results & CGM Visualizations

- **Horizontal Data Tables**: Query results render in an accessible, horizontally scrollable data table showing row and column counts (`Showing N rows × M columns`), with one-click **CSV** and **JSON** exports.
- **CGM Analytics & Charts**: Queries returning Continuous Glucose Monitoring (CGM) traces or time-series cohorts automatically compute international consensus glycaemic metrics (Time in Range `TIR 70–180 mg/dL`, `TBR <70 / <54 mg/dL`, `TAR >180 / >250 mg/dL`, `CV%`, `GMI`) and render interactive SVG charts.

---

## 🛠️ Alternate: CLI Interactive REPL

If you prefer staying completely in the terminal, you can bypass the web interface and use the built-in Interactive REPL:

```bash
t1d-analytics query --db t1d_analytics.duckdb --model gemma4 --provider ollama
```

This opens a `query>` prompt:

- If you type a raw SQL command (starting with `SELECT`, `WITH`, `SHOW`, `DESCRIBE`, `EXPLAIN`, `SUMMARIZE`, or `PRAGMA`), or set `--provider sql`, it executes directly against DuckDB.
- If you type a natural-language question, it translates it into SQL via the configured LLM provider and executes the resulting query.
- Type `exit` or `quit` to leave.
