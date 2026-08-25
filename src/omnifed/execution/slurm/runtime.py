from __future__ import annotations

import json
import os
from typing import Any, Optional

from omegaconf import OmegaConf

from .config import SlurmConfig
from .environment import build_frontier_setup_lines
from .torchdist_launcher import TorchDistSlurmLauncher
from .torchtitan_launcher import TorchTitanSlurmLauncher


class SlurmRuntime:
    """Prepare and submit an OmniFed Slurm execution."""

    def __init__(
        self,
        *,
        cfg: Any,
        hydra_cfg: Any,
        topology: Optional[Any],
        backend_name: str,
        output_dir: str,
        engine_dir: str,
        repo_root: str,
    ) -> None:
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg
        self.topology = topology
        self.backend_name = backend_name
        self.output_dir = output_dir
        self.engine_dir = engine_dir
        self.repo_root = repo_root

    def setup(self) -> None:
        if "SLURM_JOB_ID" in os.environ:
            raise RuntimeError(
                "Engine was started inside a Slurm allocation. "
                "The generated Slurm worker module should run there instead."
            )

        frozen_config_path = self._write_frozen_config()
        slurm_config = self._build_slurm_config(
            frozen_config_path
        )

        if self.backend_name == "torchdist":
            self._submit_torchdist(slurm_config)
            return

        if self.backend_name == "torchtitan":
            self._submit_torchtitan(slurm_config)
            return

        raise ValueError(
            f"Unsupported Slurm backend: {self.backend_name!r}"
        )

    @staticmethod
    def _to_container(
        value: Any,
    ) -> Any:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(
                value,
                resolve=True,
            )

        return value

    def _write_frozen_config(self) -> str:
        # Keep this file inside the unique Hydra run directory.
        # This prevents concurrent jobs from overwriting one another.
        frozen_config_path = os.path.join(
            self.output_dir,
            "engine_frozen.json",
        )

        checkpoint_dir = (
            self.cfg.slurm.checkpoint_dir
            or os.path.join(self.engine_dir, "ckpt")
        )

        frozen = {
            "cfg": OmegaConf.to_container(
                self.cfg,
                resolve=True,
            ),
            "hydra_output_dir": (
                self.hydra_cfg.runtime.output_dir
            ),
            "slurm_checkpoint_dir": checkpoint_dir,
            "backend": self._to_container(
                OmegaConf.select(
                    self.cfg,
                    "backend",
                    default={},
                )
            ),
            "torchtitan": self._to_container(
                OmegaConf.select(
                    self.cfg,
                    "torchtitan",
                    default={},
                )
            ),
        }

        with open(
            frozen_config_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                frozen,
                file,
                indent=2,
            )

        return frozen_config_path

    def _build_slurm_config(
        self,
        frozen_config_path: str,
    ) -> SlurmConfig:
        slurm_dict = OmegaConf.to_container(
            self.cfg.slurm,
            resolve=True,
        )

        if not isinstance(slurm_dict, dict):
            raise TypeError(
                "cfg.slurm must resolve to a mapping"
            )

        slurm_config = SlurmConfig(**slurm_dict)

        slurm_config.work_dir = self.repo_root
        slurm_config.cfg_json_path = frozen_config_path

        slurm_config.stdout = os.path.join(
            self.output_dir,
            "slurm-%j.out",
        )
        slurm_config.stderr = os.path.join(
            self.output_dir,
            "slurm-%j.err",
        )

        # The generated setup lines select the actual worker Python.
        slurm_config.pyexe = "python"

        configured_setup_lines = list(
            slurm_config.setup_lines
        )

        slurm_config.setup_lines = (
            build_frontier_setup_lines()
            + configured_setup_lines
        )

        return slurm_config

    def _submit_torchdist(
        self,
        slurm_config: SlurmConfig,
    ) -> None:
        if self.topology is None:
            raise RuntimeError(
                "TorchDist Slurm execution requires a topology"
            )

        total_tasks = len(self.topology)
        slurm_config.ntasks = total_tasks

        if (
            slurm_config.ntasks_per_node
            and slurm_config.ntasks_per_node > 0
        ):
            needed_nodes = (
                total_tasks
                + slurm_config.ntasks_per_node
                - 1
            ) // slurm_config.ntasks_per_node

            slurm_config.nodes = max(
                slurm_config.nodes,
                needed_nodes,
            )

        TorchDistSlurmLauncher.submit_or_exit(
            slurm_config
        )

    def _submit_torchtitan(
        self,
        slurm_config: SlurmConfig,
    ) -> None:
        subclusters = OmegaConf.select(
            self.cfg,
            "torchtitan.subclusters",
            default=None,
        )

        if subclusters is None:
            raise ValueError(
                "TorchTitan requires torchtitan.subclusters"
            )

        if not bool(
            OmegaConf.select(
                subclusters,
                "enabled",
                default=False,
            )
        ):
            raise ValueError(
                "TorchTitan Slurm execution requires "
                "torchtitan.subclusters.enabled=true"
            )

        num_clients = int(subclusters.num_clients)
        nodes_per_client = int(
            subclusters.nodes_per_client
        )
        gpus_per_node = int(
            subclusters.gpus_per_node
        )

        slurm_config.nodes = (
            1
            + num_clients
            * nodes_per_client
        )

        slurm_config.ntasks = None
        slurm_config.ntasks_per_node = gpus_per_node
        slurm_config.gpus_per_node = gpus_per_node
        slurm_config.gpus_per_task = None
        slurm_config.gres = None

        launcher_config = {
            "subclusters": OmegaConf.to_container(
                subclusters,
                resolve=True,
            ),
            "server_port": int(
                self.cfg.torchtitan.federated.server_port
            ),
        }

        TorchTitanSlurmLauncher.submit_or_exit(
            sconf=slurm_config,
            launcher_cfg=launcher_config,
        )