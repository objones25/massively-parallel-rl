"""Version pins whose bump needs a code review, checked from package metadata (no import of the library)."""

from __future__ import annotations

import importlib.metadata


def test_rsl_rl_version_is_the_one_learn_for_mirrors() -> None:
    assert importlib.metadata.version("rsl-rl-lib") == "5.4.2", (
        "TimedOnPolicyRunner.learn_for mirrors OnPolicyRunner.learn from rsl-rl-lib 5.4.2 call for call; "
        "re-check it against the new rsl_rl/runners/on_policy_runner.py before bumping this pin"
    )
