"""Operating errors: conditions that can happen in a correct program (network, quota,
missing keys, bad user input). Callers handle these; programmer errors use
`negative_space.require` and crash instead."""


class MprlOperatingError(Exception):
    """Base class for every handled, expected failure in mprl."""


class ConfigMismatch(MprlOperatingError):
    """A stored run's config differs from the requested one."""


class UploadFailed(MprlOperatingError):
    """An upload (Hugging Face, W&B) failed after its bounded retries."""


class OwnPodUnknown(MprlOperatingError):
    """The id of the pod we are running on could not be determined."""


class TerminateFailed(MprlOperatingError):
    """Terminating the pod through the RunPod API failed."""


class BudgetExceeded(MprlOperatingError):
    """The planned spend would exceed the hard cap."""


class EnvDoesNotFit(MprlOperatingError):
    """An env count does not fit in device memory."""


class MissingSecret(MprlOperatingError):
    """A required API key or token is absent from the environment."""
