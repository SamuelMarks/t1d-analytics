"""Unit tests for Sphinx documentation configuration (docs_src/conf.py)."""

import importlib
import runpy
from pathlib import Path


def test_docs_conf_variables_and_extensions() -> None:
    """Test that docs_src/conf.py correctly defines Sphinx metadata, extensions, and napoleon flags."""
    conf_path = Path(__file__).parent.parent / "docs_src" / "conf.py"
    assert conf_path.exists()

    importlib.import_module("docs_src.conf")
    mod_dict = runpy.run_path(str(conf_path))

    # Core project metadata
    assert mod_dict["project"] == "T1D Analytics"
    assert mod_dict["author"] == "Samuel Marks"
    assert "Samuel Marks" in mod_dict["copyright"]
    assert mod_dict["release"] == "0.1.0"

    # Sphinx extensions
    extensions = mod_dict["extensions"]
    assert "sphinx.ext.autodoc" in extensions
    assert "sphinx.ext.napoleon" in extensions
    assert "sphinx.ext.viewcode" in extensions

    # Napoleon settings
    assert mod_dict["napoleon_google_docstring"] is True
    assert mod_dict["napoleon_numpy_docstring"] is True
    assert mod_dict["napoleon_include_init_with_doc"] is True
    assert mod_dict["napoleon_include_private_with_doc"] is False
    assert mod_dict["napoleon_use_param"] is True
    assert mod_dict["napoleon_use_rtype"] is True
    assert mod_dict["napoleon_use_ivar"] is True

    # Suppressed warnings
    suppressed = mod_dict["suppress_warnings"]
    assert "autodoc" in suppressed
    assert "docutils" in suppressed

    # Templates, paths, and theme
    assert mod_dict["templates_path"] == ["_templates"]
    assert "_build" in mod_dict["exclude_patterns"]
    assert mod_dict["html_theme"] == "alabaster"
    assert mod_dict["html_static_path"] == []

    # Verify every automodule directive in docs_src/modules.rst is importable via conf.py's sys.path
    modules_rst = (conf_path.parent / "modules.rst").read_text(encoding="utf-8")
    documented_modules = {
        line.split("::")[1].strip()
        for line in modules_rst.splitlines()
        if line.strip().startswith(".. automodule::")
    }
    src_pkg_dir = conf_path.parent.parent / "src" / "t1d_analytics"
    expected_modules = {"t1d_analytics"} | {
        f"t1d_analytics.{p.stem}"
        for p in src_pkg_dir.glob("*.py")
        if p.stem != "__init__"
    }
    assert documented_modules == expected_modules
    for mod_name in sorted(documented_modules):
        imported = importlib.import_module(mod_name)
        assert imported.__doc__ is not None and len(imported.__doc__.strip()) > 0
