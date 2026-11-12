"""Integration tests for Sphinx documentation build and module autodoc completeness."""

import importlib
import runpy
from pathlib import Path

from sphinx.cmd.build import build_main


def test_sphinx_html_build_zero_warnings_and_module_completeness(
    tmp_path: Path,
) -> None:
    """Execute a real Sphinx HTML build with -W --keep-going and verify every src module is documented."""
    conf_path = Path(__file__).resolve().parent.parent / "docs_src" / "conf.py"
    docs_src_dir = conf_path.parent
    assert conf_path.is_file()

    importlib.import_module("docs_src.conf")
    mod_dict = runpy.run_path(str(conf_path))
    suppressed = mod_dict["suppress_warnings"]
    assert "autodoc" not in suppressed
    assert "docutils" not in suppressed

    modules_rst = (docs_src_dir / "modules.rst").read_text(encoding="utf-8")
    documented_modules = {
        line.split("::")[1].strip()
        for line in modules_rst.splitlines()
        if line.strip().startswith(".. automodule::")
    }
    src_pkg_dir = docs_src_dir.parent / "src" / "t1d_analytics"
    expected_modules = {"t1d_analytics"}
    for py_file in src_pkg_dir.rglob("*.py"):
        rel = py_file.relative_to(src_pkg_dir)
        if py_file.stem == "__init__":
            if rel.parent != Path():
                expected_modules.add("t1d_analytics." + ".".join(rel.parent.parts))
        else:
            expected_modules.add(
                "t1d_analytics." + ".".join((*rel.parent.parts, py_file.stem))
            )
    assert documented_modules == expected_modules
    for mod_name in sorted(documented_modules):
        imported = importlib.import_module(mod_name)
        assert imported.__doc__ is not None and len(imported.__doc__.strip()) > 0

    html_out_dir = tmp_path / "sphinx_html"
    doctree_dir = tmp_path / "sphinx_doctrees"
    exit_code = build_main(
        [
            "-W",
            "--keep-going",
            "-q",
            "-b",
            "html",
            "-d",
            str(doctree_dir),
            str(docs_src_dir),
            str(html_out_dir),
        ]
    )
    assert exit_code == 0
    assert (html_out_dir / "index.html").is_file()
    assert (html_out_dir / "modules.html").is_file()
