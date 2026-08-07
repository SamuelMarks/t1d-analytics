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
