"""Tests for internationalization (i18n) translation and locale handling."""

import os
from unittest.mock import patch

from t1d_analytics.i18n import TRANSLATIONS, get_translator


def test_get_translator_defaults() -> None:
    """Test get_translator fallback to LANG environment variable or English."""
    with patch.dict(os.environ, {"LANG": "ja_JP.UTF-8"}):
        t = get_translator()
        assert t("Done!") == "完了！"

    with patch.dict(os.environ, {}, clear=True):
        t = get_translator()
        assert t("Done!") == "Done!"


def test_get_translator_all_languages() -> None:
    """Test translator output across all supported languages (en, ja, ar, he)."""
    t_en = get_translator("en")
    assert t_en("Done!") == "Done!"

    t_ja = get_translator("ja")
    assert t_ja("Done!") == "完了！"

    t_ar = get_translator("ar")
    assert t_ar("Done!") == "تم بنجاح!"

    t_he = get_translator("he")
    assert t_he("Done!") == "בוצע!"


def test_translation_formatting() -> None:
    """Test translation string formatting with positional and keyword arguments."""
    t_ar = get_translator("ar")
    msg = t_ar("Downloading {}...", "dataset.csv")
    assert msg == "جارٍ تنزيل dataset.csv..."

    t_he = get_translator("he")
    msg_he = t_he("Downloading {}...", "dataset.csv")
    assert msg_he == "מוריד dataset.csv..."

    # Untranslated key formatting fallback
    t_unknown = get_translator("fr")
    msg_un = t_unknown("Hello {}", "world")
    assert msg_un == "Hello world"


def test_translations_dictionary_parity() -> None:
    """Verify that all keys in ja dictionary are also present in ar and he dictionaries."""
    ja_keys = set(TRANSLATIONS["ja"].keys())
    ar_keys = set(TRANSLATIONS["ar"].keys())
    he_keys = set(TRANSLATIONS["he"].keys())

    missing_in_ar = ja_keys - ar_keys
    missing_in_he = ja_keys - he_keys

    assert not missing_in_ar, f"Keys missing in Arabic: {missing_in_ar}"
    assert not missing_in_he, f"Keys missing in Hebrew: {missing_in_he}"


def test_codebase_translation_keys_parity() -> None:
    """Verify that all _() translation keys in Python source files exist in translation dictionaries."""
    import ast
    from pathlib import Path

    class KeyCollector(ast.NodeVisitor):
        """AST visitor collecting string literals passed to _() calls."""

        def __init__(self) -> None:
            """Initialize the collector with an empty set of keys."""
            self.keys: set[str] = set()

        def visit_Call(self, node: ast.Call) -> None:
            """Visit Call nodes and extract Constant strings from _() calls."""
            if isinstance(node.func, ast.Name) and node.func.id == "_":
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    self.keys.add(node.args[0].value)
            self.generic_visit(node)

    src_dir = Path(__file__).parent.parent / "src" / "t1d_analytics"
    all_keys: set[str] = set()
    for py_file in src_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        collector = KeyCollector()
        collector.visit(tree)
        all_keys.update(collector.keys)

    for lang in ("ja", "ar", "he"):
        lang_dict = TRANSLATIONS[lang]
        missing = [k for k in sorted(all_keys) if k not in lang_dict]
        assert not missing, f"Language '{lang}' missing keys: {missing}"
