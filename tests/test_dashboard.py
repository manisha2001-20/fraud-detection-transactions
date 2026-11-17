"""Renders every dashboard page headlessly with Streamlit's AppTest and checks there are no exceptions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Overview", "Exploratory analysis", "Suspicious transactions", "Model evaluation", "Score new transactions"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert at.title[0].value


def test_score_page_with_synthetic_batch():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.sidebar.radio[0].set_value("Score new transactions").run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Transactions scored" for m in at.metric)
