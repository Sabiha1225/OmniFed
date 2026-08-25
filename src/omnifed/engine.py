# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import rich.repr
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf

from . import utils
from .algorithm import BaseAlgorithmConfig
from .data import DataModuleConfig
from .execution import validate_execution_combination
from .execution.ray.runtime import RayRuntime
from .execution.slurm.config import SlurmConfig
from .execution.slurm.runtime import SlurmRuntime
from .model import ModelConfig
from .utils import RequiredSetup, ResultsDisplay, print

from .topology import (
    BaseTopology,
    BaseTopologyConfig,
)

@dataclass
class RayConfig:
    """Ray cluster configuration for distributed federated learning."""

    # ─────────────────────────────────────────
    # Cluster Connection & Resource Allocation
    # ─────────────────────────────────────────

    # Cluster connection (null = auto-detect local cluster)
    # Use "ray://host:port" for remote clusters, "local" to force local
    address: Optional[str] = None

    # Resource allocation - CRITICAL for proper GPU/CPU distribution
    # null = auto-detect based on hardware, explicit numbers override detection
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None

    # Custom resources: {"accelerator_type": "V100", "high_memory": 2}
    resources: Optional[Dict[str, Any]] = None

    # ─────────────────────────────────────────
    # Memory & Performance
    # ─────────────────────────────────────────

    # Object store memory for large model sharing (null = 30% of system memory)
    object_store_memory: Optional[int] = None

    # ─────────────────────────────────────────
    # Monitoring & Development
    # ─────────────────────────────────────────

    # Essential for FL: forward all distributed node logs to main process
    log_to_driver: bool = True

    # Ray dashboard (null = auto-start if dependencies available)
    include_dashboard: Optional[bool] = None
    dashboard_host: str = "127.0.0.1"  # Use "0.0.0.0" for external access
    dashboard_port: Optional[int] = None  # null = auto-find port starting from 8265

    # Development convenience - allow multiple ray.init() calls without error
    ignore_reinit_error: bool = True

    # ─────────────────────────────────────────
    # Advanced Configuration
    # ─────────────────────────────────────────

    # Experiment isolation (null = anonymous namespace)
    namespace: Optional[str] = None

    # Runtime environment for distributed workers (empty = inherit from main process)
    # Example: {"pip": ["torch==1.12.0"], "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"}}
    runtime_env: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.resources is None:
            self.resources = {}
        if self.runtime_env is None:
            self.runtime_env = {}

@dataclass
class EngineConfig:
    """Main configuration for OmniFed federated learning experiments."""

    # Required experiment parameters
    global_rounds: int = MISSING

    # Optional experiment parameters
    overwrite: bool = False

    # Component configurations - these will be resolved by Hydra defaults
    topology: BaseTopologyConfig = MISSING
    algorithm: BaseAlgorithmConfig = MISSING
    model: ModelConfig = MISSING
    datamodule: DataModuleConfig = MISSING

    # Infrastructure configurations
    ray: RayConfig = field(default_factory=RayConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    engine: Dict[str, str] = field(default_factory=lambda: {"mode": "ray"})  # 'ray' or 'slurm'

# Register the config with Hydra's ConfigStore for structured configs
cs = ConfigStore.instance()
cs.store(name="base_config", node=EngineConfig)

@rich.repr.auto
class Engine(RequiredSetup):
    """
    Main engine for federated learning experiments.

    Coordinates distributed Ray actors (nodes) to run FL algorithms across different topologies.
    Handles experiment setup, execution, and results collection with automatic
    GPU allocation and output management.

    Use this as the main entry point for running FL experiments with Hydra configurations.
    See working examples in the conf/ directory.
    """

    def __init__(
        self,
        cfg: EngineConfig,
    ) -> None:
        super().__init__()
        utils.print_rule()

        self.cfg = cfg
        self.hydra_cfg = HydraConfig.get()

        self.execution_mode = self._get_execution_mode()
        self.backend_name = self._get_backend_name()

        validate_execution_combination(
            execution_mode=self.execution_mode,
            backend_name=self.backend_name,
        )

        self.global_rounds = int(cfg.global_rounds)
        self.overwrite = bool(cfg.overwrite)

        self.output_dir = (
            self.hydra_cfg.runtime.output_dir
        )
        self.engine_dir = os.path.join(
            self.output_dir,
            "engine",
        )
        self.results_dir = os.path.join(
            self.engine_dir,
            "node_results",
        )

        self.repo_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
            )
        )

        self.topology: Optional[BaseTopology] = (
            self._create_topology()
        )

        self._results_display = ResultsDisplay()
        self.ray_runtime: Optional[RayRuntime] = None
        self.slurm_runtime: Optional[SlurmRuntime] = None

    def _get_execution_mode(self) -> str:
        return str(
            OmegaConf.select(
                self.cfg,
                "engine.mode",
                default="ray",
            )
        ).lower()

    def _get_backend_name(self) -> str:
        return str(
            OmegaConf.select(
                self.cfg,
                "backend.internal_backend",
                default="torchdist",
            )
        ).lower()

    def _create_topology(
        self,
    ) -> Optional[BaseTopology]:
        if self.backend_name != "torchdist":
            return None

        topology_config = OmegaConf.select(
            self.cfg,
            "topology",
            default=None,
        )

        if topology_config is None:
            raise ValueError(
                "TorchDist execution requires a topology"
            )

        topology = instantiate(
            topology_config,
            _recursive_=False,
        )

        if not isinstance(topology, BaseTopology):
            raise TypeError(
                "Configured topology must be an instance of "
                f"BaseTopology; got {type(topology).__name__}"
            )

        return topology

    def _setup_topology(self) -> None:
        if self.backend_name != "torchdist":
            return

        if self.topology is None:
            raise RuntimeError(
                "TorchDist execution requires a topology"
            )

        self.topology.setup(
            default_algorithm_cfg=self.cfg.algorithm,
            default_model_cfg=self.cfg.model,
            default_datamodule_cfg=self.cfg.datamodule,
        )

    def _setup_output_directories(self) -> None:
        """
        Create and validate output directories for experiment data.

        Creates engine/ and node_results/ directories under Hydra's output path.
        Issues warnings if conflicting experiment files already exist unless overwrite=True.
        """
        # Check for pre-existing files that could overwrite results (ignore Hydra standard files)
        if os.path.exists(self.output_dir):
            hydra_standard_files = {".hydra", "main.log", ".gitignore"}
            existing_files = [
                f
                for f in os.listdir(self.output_dir)
                if not f.startswith(".") and f not in hydra_standard_files
            ]
            if existing_files:
                if self.overwrite:
                    warnings.warn(
                        f"Output directory contains existing files: {self.output_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"Proceeding with overwrite=True - previous experiment results may be overwritten.",
                        UserWarning,
                    )
                else:
                    raise RuntimeError(
                        f"Output directory contains existing files: {self.output_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"This could overwrite previous experiment results. "
                        f"Use a fresh Hydra output directory, clean the existing one, or set overwrite=true."
                    )

        # Create engine directory
        os.makedirs(self.engine_dir, exist_ok=True)

        # Check if engine directory is not empty (indicates conflicting experiment)
        if os.path.exists(self.engine_dir):
            existing_files = [
                f for f in os.listdir(self.engine_dir) if not f.startswith(".")
            ]
            if existing_files:
                if self.overwrite:
                    warnings.warn(
                        f"Engine directory is not empty: {self.engine_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"Proceeding with overwrite=True - conflicting experiment files may be overwritten.",
                        UserWarning,
                    )
                else:
                    raise RuntimeError(
                        f"Engine directory is not empty: {self.engine_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"This indicates a conflicting experiment setup. Set overwrite=true to proceed anyway."
                    )

        print(f"Created engine directory: {self.engine_dir}")

    def _setup(self) -> None:
        utils.print_rule()

        self._setup_output_directories()
        self._setup_topology()

        os.environ["OMNIFED_INTERNAL_BACKEND"] = (
            self.backend_name
        )

        if self.execution_mode == "ray":
            self._setup_ray()
            return

        if self.execution_mode == "slurm":
            self._setup_slurm()
            return

        raise ValueError(
            "Unsupported execution mode: "
            f"{self.execution_mode!r}"
        )

    def _setup_ray(self) -> None:
        if self.topology is None:
            raise RuntimeError(
                "Ray execution requires an OmniFed topology"
            )

        self.ray_runtime = RayRuntime(
            cfg=self.cfg,
            hydra_cfg=self.hydra_cfg,
            topology=self.topology,
            results_display=self._results_display,
        )

        self.ray_runtime.setup()

    def _setup_slurm(self) -> None:
        self.slurm_runtime = SlurmRuntime(
            cfg=self.cfg,
            hydra_cfg=self.hydra_cfg,
            topology=self.topology,
            backend_name=self.backend_name,
            output_dir=self.output_dir,
            engine_dir=self.engine_dir,
            repo_root=self.repo_root,
        )

        self.slurm_runtime.setup()
        
    def run_experiment(self) -> None:
        """
        Start experiment execution after Engine setup.

        Ray execution remains in the parent process, so training is
        delegated to RayRuntime here.

        Slurm submission exits during setup(), and training is later
        performed by the generated Slurm worker.
        """
        if self.execution_mode != "ray":
            raise RuntimeError(
                "run_experiment() is only reached for Ray execution. "
                "Slurm submission should exit during setup()."
            )

        if self.ray_runtime is None:
            raise RuntimeError(
                "Ray runtime is not initialized. "
                "Call engine.setup() first."
            )

        self.ray_runtime.run_experiment()
