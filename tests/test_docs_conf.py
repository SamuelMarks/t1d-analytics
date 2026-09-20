"""Unit tests for Sphinx documentation configuration (docs_src/conf.py)."""

import runpy
from pathlib import Path


def test_docs_conf_variables_and_extensions() -> None:
    """Test that docs_src/conf.py correctly defines Sphinx metadata, extensions, and napoleon flags."""
    conf_path = Path(__file__).parent.parent / "docs_src" / "conf.py"
    assert conf_path.exists()

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
