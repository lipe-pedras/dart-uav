"""Name-to-class registries for every architectural axis.

Design principle 1 of the UML design ("configuration over code for variants"):
each architectural axis — block type, attention, neck, head — is resolved by
name through a registry, so a new implementation is a registration, never an
edit to an existing class.

Two registries exist:

``BLOCKS``
    Backbone block types. Built through :func:`build_block`, which is the one
    place the width multiplier (FR-2.5) is applied.

``COMPONENTS``
    Everything else, namespaced by *kind* (``"attention"``, ``"neck"``,
    ``"head"``, ``"activation"``), so two kinds may reuse a name without
    colliding.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Type, TypeVar

from ..errors import RegistryError

T = TypeVar("T")


class Registry:
    """A ``name -> class`` mapping with a decorator-style registration API."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, type] = {}

    def register(self, name: str, cls: type | None = None) -> Any:
        """Register ``cls`` under ``name``.

        Usable as a plain call (``registry.register("se", SEAttention)``) or as
        a decorator (``@registry.register("se")``).
        """

        def _do(target: Type[T]) -> Type[T]:
            key = name.lower()
            if key in self._items and self._items[key] is not target:
                raise RegistryError(
                    f"{self.kind} '{key}' is already registered to "
                    f"{self._items[key].__name__}; pick another name."
                )
            self._items[key] = target
            return target

        if cls is None:
            return _do
        return _do(cls)

    def get(self, name: str) -> type:
        key = str(name).lower()
        if key not in self._items:
            raise RegistryError(f"unknown {self.kind} '{name}'. Registered: {sorted(self._items)}")
        return self._items[key]

    def names(self) -> Iterable[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return str(name).lower() in self._items


class ComponentRegistry:
    """A registry of registries, keyed by component kind."""

    def __init__(self) -> None:
        self._kinds: Dict[str, Registry] = {}

    def kind(self, kind: str) -> Registry:
        return self._kinds.setdefault(kind, Registry(kind))

    def register(self, kind: str, name: str, cls: type | None = None) -> Any:
        return self.kind(kind).register(name, cls)

    def get(self, kind: str, name: str) -> type:
        return self.kind(kind).get(name)

    def build(self, kind: str, name: str, **kwargs: Any) -> Any:
        """Instantiate the class registered as ``(kind, name)`` with ``kwargs``."""
        return self.get(kind, name)(**kwargs)

    def names(self, kind: str) -> Iterable[str]:
        return self.kind(kind).names()


#: Backbone block types (``repconv``, ``inverted_res``, ``ghost``, ``dw_sep``).
BLOCKS = Registry("block")

#: Attention, neck, head, and activation implementations.
COMPONENTS = ComponentRegistry()


def scale_channels(channels: int, width_mult: float, minimum: int = 8) -> int:
    """Apply the backbone width multiplier to a channel count (FR-2.5).

    Mirrors the research codebase's ``ch()`` helper exactly, including the
    floor of 8 channels, so ``width_mult=1.0`` reproduces the paper's widths
    (16 -> 32 -> 48 -> 64) bit for bit.
    """
    return max(minimum, int(channels * width_mult))


def build_block(spec: Any, in_ch: int, width_mult: float = 1.0) -> Any:
    """Build one backbone block from a :class:`~dart.config.schema.BlockSpec`.

    The width multiplier is applied here and nowhere else: block classes take
    already-resolved channel counts, so they never need to know that a width
    family exists.
    """
    cls = BLOCKS.get(spec.type)
    out_ch = scale_channels(spec.out_ch, width_mult)
    kwargs: Dict[str, Any] = dict(spec.extra or {})
    return cls(
        in_ch=in_ch,
        out_ch=out_ch,
        stride=spec.stride,
        attention=spec.attention,
        act=spec.act,
        **kwargs,
    )


def register_block(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator registering a backbone block type."""
    return BLOCKS.register(name)


def register_component(kind: str, name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator registering an attention / neck / head implementation."""
    return COMPONENTS.register(kind, name)
