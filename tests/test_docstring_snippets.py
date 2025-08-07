import pytest
import traceback
from collect_doc_snippets import run_code_snippet


def test_docstring_snippet_runs(code_snippet):
    """Each code snippet from a docstring should run without error."""
    snippet_id, code = code_snippet

    try:
        run_code_snippet(code)
    except Exception:
        pytest.fail(f"Snippet failed: {snippet_id}\n\n{traceback.format_exc()}")


"""
if __name__ == "__main__":
    snippets = gather_docstring_snippets()
    for snippet_id, code in snippets:
        print(snippet_id)
        print(code)
        print("-" * 100)
        test_docstring_snippet_runs(snippet_id, code)
"""
