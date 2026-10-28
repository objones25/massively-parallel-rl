from __future__ import annotations

import pytest

from mprl import errors
from mprl.negative_space import CheckFailed, require

OPERATING_ERRORS = (
    errors.ConfigMismatch,
    errors.UploadFailed,
    errors.OwnPodUnknown,
    errors.TerminateFailed,
    errors.BudgetExceeded,
    errors.EnvDoesNotFit,
    errors.MissingSecret,
)


@pytest.mark.parametrize("error_cls", OPERATING_ERRORS, ids=lambda cls: cls.__name__)
def test_operating_error_derives_from_mprl_operating_error(error_cls: type[Exception]) -> None:
    assert issubclass(error_cls, errors.MprlOperatingError)


def test_operating_error_is_not_an_assertion_error() -> None:
    assert not issubclass(errors.MprlOperatingError, AssertionError)


def test_require_raises_check_failed_which_is_an_assertion_error() -> None:
    with pytest.raises(CheckFailed, match="num_envs must be positive") as excinfo:
        require(False, "num_envs must be positive")

    assert isinstance(excinfo.value, AssertionError)
