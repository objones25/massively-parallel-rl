"""Unit tests for ``mprl.train.git_sha``: HEAD read from the ``.git`` directory, never by spawning git.

Each test hand-builds the few files git itself would read (HEAD, a loose ref, packed-refs, a worktree's
``.git`` file and ``commondir``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mprl.train import git_sha

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def _git_dir(repo: Path, head: str) -> Path:
    git = repo / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(head)
    return git


def _loose_ref(git: Path, ref: str, sha: str) -> None:
    path = git / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{sha}\n")


def test_branch_head_reads_the_loose_ref(tmp_path: Path) -> None:
    git = _git_dir(tmp_path, "ref: refs/heads/main\n")
    _loose_ref(git, "refs/heads/main", SHA)

    assert git_sha(tmp_path) == SHA


def test_branch_head_falls_back_to_packed_refs(tmp_path: Path) -> None:
    git = _git_dir(tmp_path, "ref: refs/heads/feat/x\n")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted \n"
        f"{OTHER} refs/heads/main\n"
        f"{SHA} refs/heads/feat/x\n"
        f"{OTHER} refs/tags/v1\n"
        f"^{OTHER}\n"
    )

    assert git_sha(tmp_path) == SHA


def test_loose_ref_overrides_packed_refs(tmp_path: Path) -> None:
    git = _git_dir(tmp_path, "ref: refs/heads/main\n")
    (git / "packed-refs").write_text(f"{OTHER} refs/heads/main\n")
    _loose_ref(git, "refs/heads/main", SHA)

    assert git_sha(tmp_path) == SHA


def test_detached_head_is_the_sha_itself(tmp_path: Path) -> None:
    _git_dir(tmp_path, f"{SHA}\n")

    assert git_sha(tmp_path) == SHA


def test_worktree_git_file_follows_gitdir_and_commondir(tmp_path: Path) -> None:
    common = tmp_path / "main" / ".git"
    worktree_git = common / "worktrees" / "wt"
    worktree_git.mkdir(parents=True)
    (worktree_git / "HEAD").write_text("ref: refs/heads/wt-branch\n")
    (worktree_git / "commondir").write_text("../..\n")
    _loose_ref(common, "refs/heads/wt-branch", SHA)
    checkout = tmp_path / "wt"
    checkout.mkdir()
    (checkout / ".git").write_text(f"gitdir: {worktree_git}\n")

    assert git_sha(checkout) == SHA


@pytest.mark.parametrize(
    "head",
    [
        "ref: refs/heads/gone\n",  # the branch has no loose ref and no packed-refs entry
        "not a ref or a sha\n",
        "",
    ],
)
def test_unresolvable_head_is_unknown(tmp_path: Path, head: str) -> None:
    _git_dir(tmp_path, head)

    assert git_sha(tmp_path) == "unknown"


def test_no_checkout_is_unknown(tmp_path: Path) -> None:
    assert git_sha(tmp_path) == "unknown"


def test_malformed_ref_content_is_unknown(tmp_path: Path) -> None:
    git = _git_dir(tmp_path, "ref: refs/heads/main\n")
    _loose_ref(git, "refs/heads/main", "xyz")

    assert git_sha(tmp_path) == "unknown"


def test_never_spawns_a_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_process(*args: object, **kwargs: object) -> None:
        raise AssertionError("git_sha spawned a process")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    git = _git_dir(tmp_path, "ref: refs/heads/main\n")
    _loose_ref(git, "refs/heads/main", SHA)

    assert git_sha(tmp_path) == SHA

