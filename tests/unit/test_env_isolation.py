"""The root conftest's env isolation: every tier loses the app's variables, except that a test marked ``network``
keeps the service credentials (``WANDB_*``, ``HF_*``, ``RUNPOD_*``) it needs to reach the real services.

``support.app_env_to_strip`` is the fixture's whole decision; these tests give it this test's own node, once as
collected and once with the ``network`` marker added, so the marker lookup is the real one.
"""

from __future__ import annotations

import pytest
from support import app_env_to_strip

KEYS = ("WANDB_API_KEY", "HF_TOKEN", "RUNPOD_API_KEY", "RUNPOD_POD_ID", "MPRL_VOLUME_ROOT", "PATH", "HOME")


def test_non_network_test_strips_every_app_variable(request: pytest.FixtureRequest) -> None:
    assert app_env_to_strip(KEYS, request.node) == [
        "WANDB_API_KEY",
        "HF_TOKEN",
        "RUNPOD_API_KEY",
        "RUNPOD_POD_ID",
        "MPRL_VOLUME_ROOT",
    ]


def test_network_test_keeps_service_credentials(request: pytest.FixtureRequest) -> None:
    request.node.add_marker(pytest.mark.network)

    assert app_env_to_strip(KEYS, request.node) == ["MPRL_VOLUME_ROOT"]

