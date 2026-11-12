"""Tests for internationalization (i18n) translation and locale handling."""

import os
from unittest.mock import patch

from t1d_analytics.i18n import TRANSLATIONS, get_translator


def test_get_translator_defaults() -> None:
    """Test get_translator explicit locale resolution and independence from os.environ['LANG']."""
    with patch.dict(os.environ, {"LANG": "ja_JP.UTF-8"}):
        t_default = get_translator()
        assert t_default("Done!") == "Done!"

    t_explicit_posix = get_translator("ja_JP.UTF-8")
    assert t_explicit_posix("Done!") == "完了！"

    t_blank = get_translator("   ")
    assert t_blank("Done!") == "Done!"


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
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_"
                and node.args
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


def test_backend_error_codes_frontend_i18n_parity() -> None:
    """Verify every `backend.*` error_code in `errors.py` is translated in all 4 locales in `t1d-analytics-web/src/i18n.ts`."""
    import inspect
    import re
    from pathlib import Path

    from t1d_analytics import errors as errors_module
    from t1d_analytics.errors import T1DError

    backend_codes: set[str] = set()
    for _, cls in inspect.getmembers(errors_module, inspect.isclass):
        if issubclass(cls, T1DError):
            code = cls.default_error_code
            assert code.startswith("backend."), f"Unexpected prefix on {cls}: {code}"
            backend_codes.add(code.removeprefix("backend."))

    i18n_ts_path = (
        Path(__file__).resolve().parents[2] / "t1d-analytics-web" / "src" / "i18n.ts"
    )
    ts_source = i18n_ts_path.read_text(encoding="utf-8")
    backend_blocks = re.findall(
        r"\bbackend:\s*\{(.*?)\n  \},",
        ts_source,
        flags=re.DOTALL,
    )
    assert len(backend_blocks) == 4, (
        f"Expected 4 locale backend blocks in i18n.ts, found {len(backend_blocks)}"
    )
    locale_names = ("en", "ja", "ar", "he")
    for locale_name, block in zip(locale_names, backend_blocks, strict=True):
        present_keys = set(re.findall(r"^\s{4}(\w+):", block, flags=re.MULTILINE))
        missing_keys = sorted(backend_codes - present_keys)
        assert not missing_keys, (
            f"Frontend i18n.ts locale '{locale_name}' missing backend error keys: {missing_keys}"
        )
