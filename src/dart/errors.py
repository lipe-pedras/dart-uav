"""
Exception hierarchy for the DART framework.

Every error DART raises deliberately (as opposed to a bug) descends from
:class:`DARTError`, so a caller can catch the whole family with one clause.

Examples
--------
>>> from dart import DART
>>> from dart.errors import DARTError
>>> try:  # doctest: +SKIP
...     DART("dart-xd").train(data="datasets/my-dataset")
... except DARTError as exc:
...     print(f"DART refused the run: {exc}")
"""

from __future__ import annotations


class DARTError(Exception):
    """
    Base class for every error raised deliberately by DART.

    Catching this catches the whole family. Anything that escapes DART *without*
    descending from it is a bug, not a refusal.

    See Also
    --------
    ConfigurationError : A configuration is unusable.
    DatasetError : A dataset is unreadable or its layout is not understood.
    ExportError : An export backend failed.
    CheckpointError : A checkpoint could not be resolved or matched.
    SpawnSafetyError : Multiprocessing was requested unsafely.
    """


class ConfigurationError(DARTError):
    """
    A configuration is missing, malformed, or internally inconsistent.

    Also raised by the dataset validator when a dataset conflicts with the model
    configuration and no explicit policy resolves the conflict — the message
    always names the offending item and the config key that fixes it.
    """


class RegistryError(DARTError):
    """
    A component name was requested that is not registered.

    Raised by :class:`~dart.models.registry.Registry` and
    :class:`~dart.models.registry.ComponentRegistry`, both on lookup of an
    unknown name and on an attempt to register a name already bound to a
    different class. The message lists the registered names.
    """


class DatasetError(DARTError):
    """
    A dataset could not be read, or its layout is not understood.

    Raised by the format adapters in :mod:`dart.data.adapters` when no adapter
    recognises the directory, and by :class:`~dart.data.loader.DatasetLoader`
    when a recognised layout is missing a split it was asked for.
    """


class ExportError(DARTError):
    """
    An export backend failed to produce an artifact.

    See Also
    --------
    ExportVerificationError : The artifact was produced but does not agree with
        the PyTorch model.
    """


class ExportVerificationError(ExportError):
    """
    An artifact was produced but its outputs diverge from the PyTorch model.

    Distinct from :class:`ExportError` because the failure is numerical rather
    than structural: the file on disk is well-formed and loadable, but running
    it does not reproduce the source model within tolerance. A caller that
    wants the artifact anyway can catch this specifically.
    """


class CheckpointError(DARTError):
    """
    A checkpoint could not be resolved, read, or matched to a model.

    Covers a path that does not exist locally and cannot be fetched from the
    Hugging Face Hub, a file that loads but lacks the ``config`` and
    ``model_state`` keys every DART checkpoint carries, and a state dict whose
    tensors do not fit the architecture it is being loaded into.
    """


class SpawnSafetyError(DARTError):
    """
    Multiprocessing data loading was requested from an unguarded script.

    On Windows (and macOS under the ``spawn`` start method) a script that
    launches ``DataLoader`` workers without an ``if __name__ == "__main__":``
    guard re-executes itself in every worker, spawning processes without bound.
    DART detects the condition and raises this instead.

    Notes
    -----
    The message names both fixes: add the guard, or set ``workers=0`` to load
    data in the parent process.

    See Also
    --------
    :func:`~dart.data.loader.check_spawn_safety` : Where the condition is detected.
    """
