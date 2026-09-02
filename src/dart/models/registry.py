"""
Name-to-class registries for every architectural axis.

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
    """
    A ``name -> class`` mapping with a decorator-style registration API.

    Names are compared case-insensitively: a class registered as ``"RepConv"``
    is retrievable as ``"repconv"``, so config files are not sensitive to
    capitalisation.

    Parameters
    ----------
    kind : str
        What this registry holds, used only to phrase error messages — for
        example ``"block"`` produces "unknown block 'foo'".

    Attributes
    ----------
    kind : str
        The registry's kind, as passed to the constructor.

    See Also
    --------
    ComponentRegistry : A registry of registries, keyed by kind.
    """

    def __init__(self, kind: str) -> None:
        """Create an empty registry labelled ``kind``."""
        self.kind = kind
        self._items: Dict[str, type] = {}

    def register(self, name: str, cls: type | None = None) -> Any:
        """
        Register ``cls`` under ``name``.

        Usable as a plain call (``registry.register("se", SEAttention)``) or as
        a decorator (``@registry.register("se")``).

        Parameters
        ----------
        name : str
            Name to bind. Lowercased before storing.
        cls : type or None, default: None
            The class to bind. When ``None``, a decorator is returned instead,
            which binds whatever class it is applied to.

        Returns
        -------
        type or callable
            ``cls`` itself when it was given, so a decorator usage leaves the
            decorated class unchanged; otherwise the decorator that performs
            the binding.

        Raises
        ------
        ~dart.errors.RegistryError
            If ``name`` is already bound to a *different* class. Re-registering
            the same class under the same name is allowed, so a module that is
            imported twice does not fail.
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
        """
        Look up the class registered under ``name``.

        Parameters
        ----------
        name : str
            Registered name. Coerced to :class:`str` and lowercased, so a
            non-string config value is reported rather than missed.

        Returns
        -------
        type
            The registered class, uninstantiated.

        Raises
        ------
        ~dart.errors.RegistryError
            If ``name`` is not registered. The message lists every name that is.
        """
        key = str(name).lower()
        if key not in self._items:
            raise RegistryError(f"unknown {self.kind} '{name}'. Registered: {sorted(self._items)}")
        return self._items[key]

    def names(self) -> Iterable[str]:
        """
        List every registered name.

        Returns
        -------
        list of str
            The names, sorted, so callers rendering them to a user get a stable
            order regardless of import sequence.
        """
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        """
        Report whether ``name`` is registered.

        Parameters
        ----------
        name : object
            Candidate name; coerced to :class:`str` and lowercased, so this
            never raises on an unexpected type.

        Returns
        -------
        bool
            ``True`` if a class is bound to that name.
        """
        return str(name).lower() in self._items


class ComponentRegistry:
    """
    A registry of registries, keyed by component kind.

    Lets ``"identity"`` mean one thing as a neck and another as an attention
    module, without either registration having to qualify its name. Kinds are
    created on demand, so registering into a new kind needs no setup.

    See Also
    --------
    Registry : The per-kind mapping this delegates to.
    """

    def __init__(self) -> None:
        """Create a registry holding no kinds yet."""
        self._kinds: Dict[str, Registry] = {}

    def kind(self, kind: str) -> Registry:
        """
        Return the registry for ``kind``, creating it if needed.

        Parameters
        ----------
        kind : str
            Component kind, such as ``"attention"``, ``"neck"``, ``"head"`` or
            ``"activation"``.

        Returns
        -------
        Registry
            The registry for that kind. Always the same object for the same
            kind, so callers may hold on to it.
        """
        return self._kinds.setdefault(kind, Registry(kind))

    def register(self, kind: str, name: str, cls: type | None = None) -> Any:
        """
        Register ``cls`` as ``name`` within ``kind``.

        Parameters
        ----------
        kind : str
            Component kind. Created if it does not exist yet.
        name : str
            Name to bind within that kind.
        cls : type or None, default: None
            The class to bind, or ``None`` to return a decorator.

        Returns
        -------
        type or callable
            As :meth:`Registry.register`.

        Raises
        ------
        ~dart.errors.RegistryError
            If ``name`` is already bound to a different class in this kind.
        """
        return self.kind(kind).register(name, cls)

    def get(self, kind: str, name: str) -> type:
        """
        Look up the class registered as ``(kind, name)``.

        Parameters
        ----------
        kind : str
            Component kind.
        name : str
            Registered name within that kind.

        Returns
        -------
        type
            The registered class, uninstantiated.

        Raises
        ------
        ~dart.errors.RegistryError
            If ``name`` is not registered in that kind. Asking for an entirely
            unknown kind creates it empty, so the error names the unknown
            component rather than the unknown kind.
        """
        return self.kind(kind).get(name)

    def build(self, kind: str, name: str, **kwargs: Any) -> Any:
        """
        Instantiate the class registered as ``(kind, name)`` with ``kwargs``.

        Parameters
        ----------
        kind : str
            Component kind.
        name : str
            Registered name within that kind.
        **kwargs : Any
            Forwarded verbatim to the class constructor.

        Returns
        -------
        Any
            The constructed component.

        Raises
        ------
        ~dart.errors.RegistryError
            If ``name`` is not registered in that kind.
        """
        return self.get(kind, name)(**kwargs)

    def names(self, kind: str) -> Iterable[str]:
        """
        List every name registered within ``kind``.

        Parameters
        ----------
        kind : str
            Component kind. An unknown kind yields an empty list rather than
            raising.

        Returns
        -------
        list of str
            The names, sorted.
        """
        return self.kind(kind).names()


#: Backbone block types (``repconv``, ``inverted_res``, ``ghost``, ``dw_sep``).
BLOCKS = Registry("block")

#: Attention, neck, head, and activation implementations.
COMPONENTS = ComponentRegistry()


def scale_channels(channels: int, width_mult: float, minimum: int = 8) -> int:
    """
    Apply the backbone width multiplier to a channel count (FR-2.5).

    Mirrors the research codebase's ``ch()`` helper exactly, including the
    floor of 8 channels, so ``width_mult=1.0`` reproduces the paper's widths
    (16 -> 32 -> 48 -> 64) bit for bit.

    Parameters
    ----------
    channels : int
        Unscaled channel count, as written in the config.
    width_mult : float
        Multiplier applied to every block in the backbone.
    minimum : int, default: 8
        Floor on the result, so aggressive narrowing cannot produce a layer too
        thin to train.

    Returns
    -------
    int
        The scaled count, truncated toward zero and floored at ``minimum``.

    Notes
    -----
    Truncation, not rounding: this reproduces the research codebase's behaviour
    exactly, and changing it would silently move every published width.

    Examples
    --------
    >>> scale_channels(64, 0.5)
    32
    >>> scale_channels(8, 0.25)
    8
    """
    return max(minimum, int(channels * width_mult))


def build_block(spec: Any, in_ch: int, width_mult: float = 1.0) -> Any:
    """
    Build one backbone block from a :class:`~dart.config.schema.BlockSpec`.

    The width multiplier is applied here and nowhere else: block classes take
    already-resolved channel counts, so they never need to know that a width
    family exists.

    Parameters
    ----------
    spec : ~dart.config.schema.BlockSpec
        The block's configuration: type, unscaled output channels, stride,
        attention and activation names, plus any block-specific ``extra``
        options.
    in_ch : int
        Input channels, already scaled — normally the previous block's output.
    width_mult : float, default: 1.0
        Multiplier applied to ``spec.out_ch`` via :func:`scale_channels`.

    Returns
    -------
    ~dart.models.blocks.BackboneBlock
        The constructed block.

    Raises
    ------
    ~dart.errors.RegistryError
        If ``spec.type`` names an unregistered block type.

    See Also
    --------
    scale_channels : The width-multiplier rule applied here.
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
    """
    Return a decorator registering a backbone block type.

    Parameters
    ----------
    name : str
        Name the block becomes available under in a config's ``block.type``.

    Returns
    -------
    callable
        A decorator that registers the class it is applied to and returns it
        unchanged.

    Examples
    --------
    >>> @register_block("my_block")  # doctest: +SKIP
    ... class MyBlock(BackboneBlock):
    ...     pass
    """
    return BLOCKS.register(name)


def register_component(kind: str, name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Return a decorator registering an attention, neck, head or activation.

    Parameters
    ----------
    kind : str
        Component kind, such as ``"attention"``, ``"neck"``, ``"head"`` or
        ``"activation"``.
    name : str
        Name the component becomes available under within that kind.

    Returns
    -------
    callable
        A decorator that registers the class it is applied to and returns it
        unchanged.

    See Also
    --------
    register_block : The equivalent for backbone block types.
    """
    return COMPONENTS.register(kind, name)
