from __future__ import annotations

import pytest

from autotree_sdk import TreeClient

from .mock_asgi import MockAutoTreeASGI, make_http_client


@pytest.fixture
def mock_app() -> MockAutoTreeASGI:
    return MockAutoTreeASGI()


@pytest.fixture
def tree_client(mock_app: MockAutoTreeASGI):
    http_client = make_http_client(mock_app)
    client = TreeClient("http://autotree.test", http_client=http_client)
    try:
        yield client
    finally:
        http_client.close()
