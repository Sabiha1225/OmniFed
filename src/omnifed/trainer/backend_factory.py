from __future__ import annotations

from typing import Any


def create_backend(cfg: Any | None = None, **kwargs: Any) -> Any:
    """Create the configured local backend for OmniFed."""

    backend_name = None
    if cfg is not None:
        backend_cfg = getattr(cfg, "backend", None)
        if backend_cfg is None and isinstance(cfg, dict):
            backend_cfg = cfg.get("backend")
        if backend_cfg is not None:
            backend_name = getattr(backend_cfg, "internal_backend", None)
            if backend_name is None and isinstance(backend_cfg, dict):
                backend_name = backend_cfg.get("internal_backend")

    backend_name = backend_name or kwargs.pop("backend_name", "torchdist")

    if backend_name == "torchtitan":
        from .torchtitan_backend import TorchTitanBackend

        return TorchTitanBackend(cfg=cfg, **kwargs)

    if backend_name == "torchdist":
        from ..communicator.torchdist import TorchDistCommunicator

        return TorchDistCommunicator(**kwargs)

    raise ValueError(f"Unsupported internal backend: {backend_name}")
