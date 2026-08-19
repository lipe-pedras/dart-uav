"""Exception hierarchy for the DART framework.

Every error DART raises deliberately (as opposed to a bug) descends from
:class:`DARTError`, so a caller can catch the whole family with one clause.
"""

from __future__ import annotations


class DARTError(Exception):
    """Base class for every error raised deliberately by DART."""


class ConfigurationError(DARTError):
    """A configuration is missing, malformed, or internally inconsistent.

    Also raised by the dataset validator (FR-1.7) when a dataset conflicts with
    the model configuration and no explicit policy resolves the conflict — the
    message always names the offending item and the config key that fixes it.
    """


class RegistryError(DARTError):
    """A component name was requested that is not registered."""


class DatasetError(DARTError):
    """A dataset could not be read, or its layout is not understood."""


class ExportError(DARTError):
    """An export backend failed to produce an artifact."""


class ExportVerificationError(ExportError):
    """An artifact was produced but its outputs diverge from the PyTorch model."""


class CheckpointError(DARTError):
    """A checkpoint could not be resolved, read, or matched to a model."""


class SpawnSafetyError(DARTError):
    """Multiprocessing data loading was requested from an unguarded script.

    On Windows (and macOS under the ``spawn`` start method) a script that
    launches ``DataLoader`` workers without an ``if __name__ == "__main__":``
    guard re-executes itself in every worker, spawning processes without bound.
    DART detects the condition and raises this instead (NFR-3).
    """
