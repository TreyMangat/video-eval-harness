from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.name == "test_integration.py":
            item.add_marker(pytest.mark.integration)
