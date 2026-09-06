"""Internationalization and localization module."""

import os
from typing import Any, Callable, Optional

# A simple dictionary-based fallback if we want to provide translations in code,
# or we can just use a simple translation dict here.

TRANSLATIONS = {
    "en": {},
    "ja": {
        "An error occurred: {}": "エラーが発生しました: {}",
        "Fetching HTML from {}...": "{}からHTMLを取得中...",
        "Parsing datasets...": "データセットを解析中...",
        "Found {} protocols.": "{}個のプロトコルが見つかりました。",
        "No datasets found. Exiting.": "データセットが見つかりません。終了します。",
        "Starting downloads to {}...": "{}へのダウンロードを開始します...",
        "Done!": "完了！",
        "Data directory {} does not exist.": "データディレクトリ{}は存在しません。",
        "Checking for zip files to extract...": "解凍するzipファイルを確認中...",
        "Extracting {}...": "{}を解凍中...",
        "Failed to extract {} (Bad Zip File).": "{}の解凍に失敗しました（不正なZipファイル）。",
        "No zip files found to extract.": "解凍するzipファイルが見つかりません。",
        "Extraction complete.": "解凍が完了しました。",
        "Connecting to DuckDB at {}...": "{}のDuckDBに接続中...",
        "Scanning for CSV and TXT files...": "CSVおよびTXTファイルをスキャン中...",
        "No tabular files found.": "表形式のファイルが見つかりません。",
        "Loading {} (encoding={}, sep='{}') into table {}...": "{}（エンコーディング={}、セパレータ='{}'）をテーブル{}に読み込み中...",
        "Table {} already exists, skipping...": "テーブル{}はすでに存在します。スキップします...",
        "Failed to load {}: {}": "{}の読み込みに失敗しました: {}",
        "Successfully populated {} tables in {}.": "{}個のテーブルを{}に正常に読み込みました。",
        "Database {} does not exist.": "データベース{}は存在しません。",
        "Please run the 'load' command first to populate the database.": "データベースにデータを読み込むには、最初に'load'コマンドを実行してください。",
        "Welcome to T1D Analytics Interface!": "T1D Analytics Interfaceへようこそ！",
        "You can enter:": "以下を入力できます：",
        "  - Standard SQL queries (starting with SELECT, WITH, SHOW, DESCRIBE, etc.)": "  - 標準のSQLクエリ（SELECT、WITH、SHOW、DESCRIBEなどで始まる）",
        "  - Natural language queries (will be translated to SQL via LLM)": "  - 自然言語のクエリ（LLMによってSQLに変換されます）",
        "  - 'exit' or 'quit' to close.": "  - 終了するには'exit'または'quit'を入力します。",
        "Exiting.": "終了します。",
        "SQL Error: {}": "SQLエラー: {}",
        "Error: any-llm-sdk[ollama] is not installed. Please install it.": "エラー: any-llm-sdk[ollama]がインストールされていません。インストールしてください。",
        "Thinking...": "考え中...",
        "Generated SQL: \n{}\n": "生成されたSQL: \n{}\n",
        "Executing...\n": "実行中...\n",
        "Failed to generate or execute query: {}": "クエリの生成または実行に失敗しました: {}",
        "Saved DOI link: {}": "DOIリンクを保存しました: {}",
        "DOI link already exists, skipping: {}": "DOIリンクはすでに存在します。スキップします: {}",
        "File already exists, skipping: {}": "ファイルはすでに存在します。スキップします: {}",
        "Downloading {}...": "{}をダウンロード中...",
        "Processing protocol: {}": "プロトコルを処理中: {}",
        "Initializing TrainingDataGenerator with model '{}'...": "モデル'{}'でTrainingDataGeneratorを初期化中...",
        "Found {} tables in the schema.": "スキーマ内に{}個のテーブルが見つかりました。",
        "Generating {} pairs for table: {}...": "テーブル: {}のペアを{}個生成中...",
        "Training data generation complete!": "トレーニングデータの生成が完了しました！",
        "Database file '{}' does not exist.": "データベースファイル'{}'は存在しません。",
        "Run 't1d-analytics load --data-dir <data_dir> --db {}' to create and populate the database.": "データベースを作成して読み込むには、't1d-analytics load --data-dir <data_dir> --db {}'を実行してください。",
        "Database file '{}' is empty (0 bytes).": "データベースファイル'{}'は空（0バイト）です。",
        "Populate the database with 't1d-analytics load --data-dir <data_dir> --db {}'.": "'t1d-analytics load --data-dir <data_dir> --db {}'でデータベースにデータを読み込んでください。",
        "Cannot access database file '{}': {}": "データベースファイル'{}'にアクセスできません: {}",
        "Verify filesystem read permissions for the database file.": "データベースファイルのファイルシステム読み取り権限を確認してください。",
        "Failed to open DuckDB database '{}': {}": "DuckDBデータベース'{}'を開くことができませんでした: {}",
        "Check that the database is not corrupted or locked by another process.": "データベースが破損していないか、または別のプロセスによってロックされていないか確認してください。",
        "Database '{}' connected successfully but contains 0 tables.": "データベース'{}'に正常に接続されましたが、テーブルが0件です。",
        "Run 't1d-analytics load --data-dir <data_dir> --db {}' to load clinical trial data.": "臨床試験データを読み込むには、't1d-analytics load --data-dir <data_dir> --db {}'を実行してください。",
        "Database '{}' contains {} custom table(s), but lacks standard T1D clinical trial datasets (e.g. patients, cgms, dclp3).": "データベース'{}'には{}個のカスタムテーブルが含まれていますが、標準的なT1D臨床試験データセット（例: patients, cgms, dclp3）がありません。",
        "Download and load public clinical trial datasets using 't1d-analytics download' and 't1d-analytics load'.": "'t1d-analytics download'および't1d-analytics load'を使用して、公開臨床試験データセットをダウンロードして読み込んでください。",
        "Warning: Database {} contains 0 tables.": "警告: データベース{}にはテーブルが0件です。",
        "Warning: Database {} lacks standard initial clinical trial datasets.": "警告: データベース{}には標準的な初期臨床試験データセットが不足しています。",
        "Database '{}' is healthy with {} table(s) ready.": "データベース'{}'は正常で、{}個のテーブルが準備完了です。",
        "Ollama is accessible, but recommended model '{}' is not pulled.": "Ollamaにアクセス可能ですが、推奨モデル'{}'が取得されていません。",
        "Run 'ollama pull {}' to enable full Natural Language query translation.": "完全な自然言語クエリ翻訳を有効にするには、'ollama pull {}'を実行してください。",
        "Ollama LLM service is online and ready.": "Ollama LLMサービスはオンラインで準備完了です。",
        "Cannot connect to local Ollama LLM service at {}.": "{}にあるローカルOllama LLMサービスに接続できません。",
        "Start Ollama or select 'Literal SQL' mode in the web interface to query directly.": "直接クエリを実行するには、Ollamaを起動するか、Webインターフェースで'Literal SQL'モードを選択してください。",
    },
}


def get_translator(lang: Optional[str] = None) -> Callable[..., str]:
    """
    Get a translator function for the given language.

    Args:
    ----
        lang: The language code (e.g., "en", "ja"). If None, uses the LANG env var.

    Returns:
    -------
        A translation function.

    """
    if lang is None:
        lang = os.environ.get("LANG", "en").split(".")[0].split("_")[0]

    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])

    def translate(text: str, *args: Any, **kwargs: Any) -> str:
        """Translate the given text and format it with args and kwargs."""
        translated = lang_dict.get(text, text)
        if args or kwargs:
            return translated.format(*args, **kwargs)
        return translated

    return translate


# Default translator
_ = get_translator()
