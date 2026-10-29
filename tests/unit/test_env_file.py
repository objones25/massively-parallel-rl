"""Unit tests for mprl.env_file: parsing .env.example, rendering and writing .env, prompting."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from dotenv import dotenv_values

from mprl.env_file import EnvVar, parse_env_example, prompt_values, render_env, write_env
from mprl.errors import InvalidInput
from mprl.negative_space import CheckFailed

EXAMPLE = """\
# Header block, not attached to any key.

# Get it at https://wandb.ai/authorize
WANDB_API_KEY=

# Project name
WANDB_PROJECT=mprl-humanoid-walk

# Hugging Face token
# (write access)
HF_TOKEN=
"""

REPO_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def test_parse_env_example_marks_secrets() -> None:
    parsed = parse_env_example(EXAMPLE)

    assert parsed == [
        EnvVar("WANDB_API_KEY", None, "Get it at https://wandb.ai/authorize", secret=True),
        EnvVar("WANDB_PROJECT", "mprl-humanoid-walk", "Project name", secret=False),
        EnvVar("HF_TOKEN", None, "Hugging Face token\n(write access)", secret=True),
    ]


def test_parse_env_example_requires_a_comment_per_key() -> None:
    with pytest.raises(CheckFailed, match="NO_COMMENT"):
        parse_env_example("NO_COMMENT=1\n")


def test_repo_env_example_has_the_documented_keys() -> None:
    parsed = parse_env_example(REPO_EXAMPLE.read_text())

    assert [(v.name, v.default) for v in parsed] == [
        ("WANDB_API_KEY", None),
        ("WANDB_ENTITY", None),
        ("WANDB_PROJECT", "mprl-humanoid-walk"),
        ("HF_TOKEN", None),
        ("HF_REPO_ID", None),
        ("RUNPOD_API_KEY", None),
        ("MPRL_VOLUME_ROOT", "/workspace/mprl"),
    ]
    assert [v.name for v in parsed if v.secret] == ["WANDB_API_KEY", "HF_TOKEN", "RUNPOD_API_KEY"]


def test_render_env_preserves_order_and_comments(tmp_path: Path) -> None:
    parsed = parse_env_example(EXAMPLE)

    text = render_env(parsed, {"HF_TOKEN": "hf_abc", "WANDB_API_KEY": "k" * 40})

    assert text == (
        "# Get it at https://wandb.ai/authorize\n"
        f"WANDB_API_KEY={'k' * 40}\n"
        "\n"
        "# Project name\n"
        "WANDB_PROJECT=mprl-humanoid-walk\n"
        "\n"
        "# Hugging Face token\n"
        "# (write access)\n"
        "HF_TOKEN=hf_abc\n"
    )
    (tmp_path / ".env").write_text(text)
    assert dotenv_values(tmp_path / ".env") == {
        "WANDB_API_KEY": "k" * 40,
        "WANDB_PROJECT": "mprl-humanoid-walk",
        "HF_TOKEN": "hf_abc",
    }


@pytest.mark.parametrize("value", ["has space", "a#b", "${HOME}", "quo'te", 'dq"', "back\\slash", "new\nline"])
def test_render_env_rejects_values_dotenv_would_change(value: str) -> None:
    with pytest.raises(InvalidInput, match="HF_TOKEN"):
        render_env(parse_env_example(EXAMPLE), {"HF_TOKEN": value})


def test_write_env_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("OLD=1\n")

    with pytest.raises(FileExistsError):
        write_env(path, "NEW=2\n", overwrite=False)

    assert path.read_text() == "OLD=1\n"


def test_write_env_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / ".env"

    write_env(path, "A=1\n", overwrite=False)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text() == "A=1\n"


def test_write_env_overwrite_replaces_and_tightens_mode(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("OLD=1\n")
    path.chmod(0o644)

    write_env(path, "NEW=2\n", overwrite=True)

    assert path.read_text() == "NEW=2\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == [".env"]


def test_prompt_values_uses_getpass_for_secrets_and_keeps_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    parsed = parse_env_example(EXAMPLE)
    asked: list[tuple[str, str]] = []
    answers = {"WANDB_API_KEY": "", "WANDB_PROJECT": "", "HF_TOKEN": "hf_new"}

    def ask(prompt: str) -> str:
        asked.append(("input", prompt))
        return answers[prompt.split(" ")[0]]

    def ask_secret(prompt: str) -> str:
        asked.append(("getpass", prompt))
        return answers[prompt.split(" ")[0]]

    values = prompt_values(parsed, {"WANDB_API_KEY": "old_secret"}, ask=ask, ask_secret=ask_secret)

    assert values == {"WANDB_API_KEY": "old_secret", "WANDB_PROJECT": "mprl-humanoid-walk", "HF_TOKEN": "hf_new"}
    assert asked == [
        ("getpass", "WANDB_API_KEY [set; Enter keeps it]: "),
        ("input", "WANDB_PROJECT [mprl-humanoid-walk]: "),
        ("getpass", "HF_TOKEN [not set]: "),
    ]
    out = capsys.readouterr().out
    assert "old_secret" not in out
    assert "# Get it at https://wandb.ai/authorize" in out


def test_prompt_values_asks_again_after_an_invalid_value() -> None:
    parsed = [EnvVar("HF_REPO_ID", None, "repo", secret=False)]
    replies = iter(["bad value", "user/repo"])

    values = prompt_values(parsed, {}, ask=lambda _prompt: next(replies), ask_secret=lambda _prompt: "")

    assert values == {"HF_REPO_ID": "user/repo"}


def test_prompt_values_gives_up_after_three_invalid_values() -> None:
    parsed = [EnvVar("HF_REPO_ID", None, "repo", secret=False)]

    with pytest.raises(InvalidInput, match="HF_REPO_ID"):
        prompt_values(parsed, {}, ask=lambda _prompt: "bad value", ask_secret=lambda _prompt: "")
