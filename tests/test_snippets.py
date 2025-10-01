import pytest
import traceback


def run_code_snippet(code, context=None):
    """Run a snippet of code and catch any exceptions."""
    context = context or {}
    exec(code, context)


def test_docstring_snippet_runs(doc_snippet):
    """Each code snippet from a docstring should run without error."""
    snippet_id, code = doc_snippet

    try:
        run_code_snippet(code)
    except Exception:
        pytest.fail(f"Snippet failed: {snippet_id}\n\n{traceback.format_exc()}")


def test_markdown_snippet_runs(md_snippet):
    snippet_id, code = md_snippet
    try:
        run_code_snippet(code)
    except Exception:
        pytest.fail(f"Snippet failed: {snippet_id}\n\n{traceback.format_exc()}")
