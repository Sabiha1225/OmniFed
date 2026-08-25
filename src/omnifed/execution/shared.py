from __future__ import annotations


SUPPORTED_EXECUTION_COMBINATIONS = frozenset(
    {
        ("ray", "torchdist"),
        ("slurm", "torchdist"),
        ("slurm", "torchtitan"),
    }
)


def validate_execution_combination(
    execution_mode: str,
    backend_name: str,
) -> None:
    combination = (
        execution_mode.lower(),
        backend_name.lower(),
    )

    if combination not in SUPPORTED_EXECUTION_COMBINATIONS:
        supported = ", ".join(
            f"{mode}+{backend}"
            for mode, backend in sorted(
                SUPPORTED_EXECUTION_COMBINATIONS
            )
        )

        raise ValueError(
            "Unsupported execution configuration: "
            f"engine.mode={execution_mode!r}, "
            f"backend.internal_backend={backend_name!r}. "
            f"Supported combinations: {supported}"
        )