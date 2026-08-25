from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from dataclasses import asdict, fields, is_dataclass
from typing import Any

import ray
from omegaconf import OmegaConf
from rich.pretty import pprint
from tqdm.auto import tqdm

from ...utils import (
    ResultsDisplay,
    print,
    print_rule,
)
from .actor import RayActor


LOG_FLUSH_DELAY = 2.0


class RayRuntime:
    def __init__(
        self,
        cfg: Any,
        hydra_cfg: Any,
        topology: Any,
        results_display: ResultsDisplay,
    ) -> None:
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg
        self.topology = topology
        self.results_display = results_display

        self.ray_cfg = cfg.ray
        self.global_rounds = int(
            cfg.global_rounds
        )

        self.engine_dir = os.path.join(
            hydra_cfg.runtime.output_dir,
            "engine",
        )

        self.results_dir = os.path.join(
            self.engine_dir,
            "node_results",
        )

        self.actor_refs: list[Any] = []

    def _ray_init_config(
        self,
    ) -> dict[str, Any]:
        if is_dataclass(self.ray_cfg):
            ray_config = asdict(
                self.ray_cfg
            )
        else:
            converted = OmegaConf.to_container(
                self.ray_cfg,
                resolve=True,
            )

            if not isinstance(converted, dict):
                raise TypeError(
                    "cfg.ray must resolve to a mapping"
                )

            ray_config = converted

        address = ray_config.get(
            "address"
        )

        if address not in (
            None,
            "",
            "local",
        ):
            for key in (
                "num_cpus",
                "num_gpus",
                "resources",
                "object_store_memory",
                "include_dashboard",
                "dashboard_host",
                "dashboard_port",
                "runtime_env",
            ):
                ray_config.pop(
                    key,
                    None,
                )

        return ray_config

    def setup(self) -> None:
        ray.init(
            **self._ray_init_config()
        )

        available_resources = (
            ray.available_resources()
        )
        nodes = ray.nodes()

        print(
            "ray.available_resources()"
        )
        pprint(available_resources)

        print("ray.nodes()")
        pprint(nodes)

        resources_path = os.path.join(
            self.engine_dir,
            "ray_available_resources.json",
        )
        nodes_path = os.path.join(
            self.engine_dir,
            "ray_nodes.json",
        )

        with open(
            resources_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                available_resources,
                file,
                indent=2,
                default=str,
            )

        with open(
            nodes_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                nodes,
                file,
                indent=2,
                default=str,
            )

        alive_nodes = [
            node
            for node in nodes
            if node["Alive"]
        ]

        available_gpus = float(
            available_resources.get(
                "GPU",
                0,
            )
        )
        total_actors = len(
            self.topology
        )

        if available_gpus == 0:
            gpus_per_actor = 0.0
        elif total_actors <= available_gpus:
            gpus_per_actor = 1.0
        else:
            gpus_per_actor = max(
                available_gpus
                / total_actors,
                0.25,
            )

        print(
            f"Ray nodes alive: "
            f"{len(alive_nodes)}"
        )
        print(
            f"Available GPUs: "
            f"{available_gpus}"
        )
        print(
            f"Launching {total_actors} "
            "Ray actors"
        )

        self.actor_refs = (
            self._create_actors(
                gpus_per_actor
            )
        )

        setup_futures = [
            actor.setup.remote(
                total_rounds=(
                    self.global_rounds
                ),
            )
            for actor in self.actor_refs
        ]

        ray.get(setup_futures)

    def _create_actors(
        self,
        gpus_per_actor: float,
    ) -> list[Any]:
        actor_refs: list[Any] = []

        for node_config in self.topology:
            node_config.log_dir_base = (
                node_config.log_dir_base
                or self.hydra_cfg.runtime.output_dir
            )

            actor_options = node_config.ray_actor_options

            if OmegaConf.is_config(actor_options):
                converted_options = OmegaConf.to_container(
                    actor_options,
                    resolve=True,
                )
                if not isinstance(converted_options, dict):
                    raise TypeError(
                        "ray_actor_options must resolve to a mapping"
                    )
                actor_options_dict = converted_options
            elif is_dataclass(actor_options):
                actor_options_dict = asdict(actor_options)
            else:
                actor_options_dict = dict(actor_options)

            if actor_options_dict.get("num_gpus") is None:
                actor_options_dict["num_gpus"] = gpus_per_actor

            # Ray should not receive options whose value is None.
            actor_options_dict = {
                key: value
                for key, value in actor_options_dict.items()
                if value is not None
            }

            if OmegaConf.is_config(node_config):
                node_kwargs = {
                    key: node_config[key]
                    for key in node_config.keys()
                }
            elif is_dataclass(node_config):
                # Shallow conversion preserves Hydra component configurations.
                node_kwargs = {
                    field.name: getattr(node_config, field.name)
                    for field in fields(node_config)
                }
            else:
                node_kwargs = dict(node_config)

            print_rule()
            pprint(node_config)

            node_kwargs.pop(
                "ray_actor_options",
                None,
            )

            actor = RayActor.options(
                **actor_options_dict
            ).remote(
                **node_kwargs
            )

            actor_refs.append(actor)

        return actor_refs

    def _save_results(
        self,
        results: list[Any],
    ) -> None:
        os.makedirs(
            self.results_dir,
            exist_ok=True,
        )

        for node_index, result in tqdm(
            enumerate(results),
            desc="Saving node results",
            unit="file",
            total=len(results),
        ):
            result_path = os.path.join(
                self.results_dir,
                (
                    f"node_{node_index:03d}"
                    "_results.pkl"
                ),
            )

            with open(
                result_path,
                "wb",
            ) as file:
                pickle.dump(
                    result,
                    file,
                )

    def run_experiment(self) -> None:
        try:
            print_rule()

            start_time = time.time()

            result_futures = [
                actor.run_experiment.remote()
                for actor in self.actor_refs
            ]

            results = ray.get(
                result_futures
            )

            try:
                self._save_results(
                    results
                )
            except Exception as exception:
                warnings.warn(
                    "Failed to save individual "
                    "Ray node results: "
                    f"{exception}",
                    UserWarning,
                )

            duration = (
                time.time() - start_time
            )

            time.sleep(
                LOG_FLUSH_DELAY
            )

            self.results_display.show_experiment_results(
                results,
                duration,
                self.global_rounds,
                len(self.topology),
            )
        finally:
            print(
                "Shutting down Ray",
                flush=True,
            )
            ray.shutdown()