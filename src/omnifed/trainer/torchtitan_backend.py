from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
# from src.omnifed.communicator import AggregationOp

import torch.distributed as dist

from torch.distributed.checkpoint.format_utils import (
    dcp_to_torch_save,
    torch_save_to_dcp,
)

# from torch.distributed.checkpoint.state_dict import (
#     StateDictOptions,
#     get_model_state_dict,
# )



class TorchTitanBackend:
    """Thin adapter that lets OmniFed use Torchtitan as a local trainer.

    The adapter uses Torchtitan's registry-based config flow so OmniFed can
    select a model config by module and config name instead of requiring a
    standalone TOML file.
    """

    def __init__(self, cfg: Any | None = None, **kwargs: Any) -> None:
        self.cfg = cfg
        self._trainer: Any | None = None
        self._config: Any | None = None
        self._module = kwargs.pop("module", None)
        self._config_name = kwargs.pop("config_name", None)
        self._overrides = kwargs.pop("overrides", None) or {}
        self._output_dir = kwargs.pop("output_dir", None)
        self._update_dir = kwargs.pop("update_dir", None)

    @staticmethod
    def _get_nested(cfg: Any | None, attr: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(attr, default)
        return getattr(cfg, attr, default)

    def _resolve_config(self) -> dict[str, Any]:
        torchtitan_cfg = self._get_nested(self.cfg, "torchtitan", None)
        module = self._module or self._get_nested(torchtitan_cfg, "module", "llama3")
        config_name = self._config_name or self._get_nested(
            torchtitan_cfg, "config_name", None
        )
        overrides = self._overrides or self._get_nested(
            torchtitan_cfg, "overrides", {}
        )
        output_dir = self._output_dir or self._get_nested(
            torchtitan_cfg, "output_dir", "outputs/torchtitan_site"
        )
        update_dir = self._update_dir or self._get_nested(
            torchtitan_cfg, "update_dir", "outputs/federated_updates"
        )
        hf_assets_path = self._get_nested(
            torchtitan_cfg, "hf_assets_path", None
        )
        dataset_path = self._get_nested(
            torchtitan_cfg, "dataset_path", None
        )

        if not config_name:
            raise ValueError(
                "Torchtitan config_name is required. Set torchtitan.config_name or pass it explicitly."
            )

        return {
            "module": module,
            "config_name": config_name,
            "overrides": overrides,
            "output_dir": output_dir,
            "update_dir": update_dir,
            "root": self._get_nested(
                torchtitan_cfg, "root", os.environ.get("TORCHTITAN_ROOT")
            ),
            "hf_assets_path": hf_assets_path,
            "dataset_path": dataset_path,
        }

    def _add_torchtitan_to_path(self, root: str | os.PathLike[str] | None) -> None:
        if not root:
            return

        torchtitan_root = Path(root).expanduser()
        if not torchtitan_root.exists():
            raise FileNotFoundError(
                f"Configured torchtitan.root does not exist: {torchtitan_root}"
            )

        torchtitan_root_str = str(torchtitan_root.resolve())
        if torchtitan_root_str not in sys.path:
            sys.path.insert(0, torchtitan_root_str)

    def setup(self) -> Any:
        if self._trainer is not None:
            return self._trainer

        config_spec = self._resolve_config()
        self._add_torchtitan_to_path(config_spec["root"])

        try:
            from torchtitan.config.manager import ConfigManager
            from torchtitan.trainer import Trainer
        except Exception as exc:  # pragma: no cover - environment-dependent import
            raise RuntimeError(
                "Torchtitan is not installed or not importable. "
                "Install it under the sibling torchtitan workspace and ensure the package is on PYTHONPATH."
            ) from exc

        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        output_dir = Path(config_spec["output_dir"])

        client_id = os.environ.get("CLIENT_ID")
        job_id = os.environ.get("SLURM_JOB_ID")

        if client_id is not None:
            if job_id is not None:
                output_dir = output_dir / f"job_{job_id}"
            output_dir = output_dir / f"client_{client_id}"

        output_dir.mkdir(parents=True, exist_ok=True)

        args = [f"--module={config_spec['module']}", f"--config={config_spec['config_name']}"]
        for key, value in (config_spec["overrides"] or {}).items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value)
            args.append(f"--{key}={value}")

        config_manager = ConfigManager()
        config = config_manager.parse_args(args=args)
        if config_spec.get("hf_assets_path"):
            config.hf_assets_path = config_spec["hf_assets_path"]
        if config_spec.get("dataset_path"):
            config.dataloader.dataset_path = config_spec["dataset_path"]
        config.dump_folder = str(output_dir)
        self._config = config
        self._trainer = Trainer(config)
        self._trainer.config.dump_folder = str(output_dir)
        return self._trainer

    def train_local_steps(
        self,
        steps: int | None = None,
    ) -> dict[str, Any]:
        trainer = self.setup()

        local_steps = int(steps or 1)
        tokens_before = int(trainer.ntokens_seen)

        data_iterator = trainer.batch_generator(
            trainer.dataloader
        )

        steps_run = 0

        for _ in range(local_steps):
            trainer.step += 1
            trainer.train_step(data_iterator)
            steps_run += 1

        tokens_this_round = (
            int(trainer.ntokens_seen) - tokens_before
        )

        return {
            "steps_run": steps_run,
            "step": trainer.step,
            "num_tokens": tokens_this_round,
        }

    # def export_update(self, round_id: int | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    #     trainer = self.setup()
    #     config_spec = self._resolve_config()
    #     update_dir = Path(config_spec["update_dir"])
    #     update_dir.mkdir(parents=True, exist_ok=True)

    #     artifact_name = f"round_{round_id if round_id is not None else trainer.step}.pt"
    #     artifact_path = update_dir / artifact_name
    #     payload = {
    #         "round_id": round_id if round_id is not None else trainer.step,
    #         "step": trainer.step,
    #         "module": config_spec["module"],
    #         "config_name": config_spec["config_name"],
    #         "metadata": metadata or {},
    #         "config": trainer.config.to_dict() if hasattr(trainer.config, "to_dict") else None,
    #     }
    #     torch.save(payload, artifact_path)
    #     return {"path": str(artifact_path), "round_id": payload["round_id"], "step": payload["step"]}

    # def load_global_update(self, update_path: str | os.PathLike[str] | None = None) -> Any:
    #     if not update_path:
    #         return None
    #     path = Path(update_path)
    #     if not path.exists():
    #         raise FileNotFoundError(f"Update artifact not found: {path}")
    #     return torch.load(path, map_location="cpu")

    # def aggregate_model(self, comm, weight: float) -> None:
    #     trainer = self.setup()

    #     for model_part in trainer.model_parts:
    #         for param in model_part.parameters():
    #             if param.requires_grad:
    #                 param.data.mul_(weight)
    #                 comm.aggregate(param.data, reduction=AggregationOp.SUM)

    
    def save_and_consolidate(
        self,
        round_id: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        trainer = self.setup()

        rank = int(os.environ["RANK"])
        client_id = int(os.environ["CLIENT_ID"])
        leader_rank = int(os.environ["CLIENT_LEADER_RANK"])

        round_root = (
            Path(os.environ["CLIENT_CHECKPOINT_ROOT"])
            / f"round_{round_id}"
        )

        distributed_root = round_root / "local_dcp"
        consolidated_path = (
            round_root / "consolidated" / "model.pt"
        )

        # Redirect TorchTitan checkpointing for this round.
        original_folder = trainer.checkpointer.folder
        trainer.checkpointer.folder = str(distributed_root)

        try:
            saved = trainer.checkpointer.save(
                trainer.step,
                last_step=True,
            )

            if not saved:
                raise RuntimeError(
                    "TorchTitan did not save a checkpoint. "
                    "Set checkpoint.enable=true."
                )

            dist.barrier()

            dcp_path = (
                distributed_root
                / f"step-{trainer.step}"
            )

            # DCP conversion collects all PP/TP parameter shards.
            if rank == leader_rank:
                consolidated_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                temporary_flat_path = (
                    consolidated_path.parent / "flat_model.tmp.pt"
                )

                dcp_to_torch_save(
                    str(dcp_path),
                    str(temporary_flat_path),
                )

                flat_state = torch.load(
                    temporary_flat_path,
                    map_location="cpu",
                    weights_only=False,
                )

                temporary_path = consolidated_path.with_suffix(
                    ".tmp"
                )

                torch.save(
                    {
                        "model": flat_state,
                        "client_id": client_id,
                        "round_id": round_id,
                        "step": trainer.step,
                        "num_tokens": int(num_tokens),
                    },
                    temporary_path,
                )

                os.replace(temporary_path, consolidated_path)
                temporary_flat_path.unlink(missing_ok=True)

            dist.barrier()

        finally:
            trainer.checkpointer.folder = original_folder

        return {
            "client_id": client_id,
            "round_id": round_id,
            "checkpoint_path": str(consolidated_path),
            "num_tokens": int(num_tokens),
            "is_leader": rank == leader_rank,
    }

    def load_global_model(
        self,
        model_path: str | os.PathLike[str],
        round_id: int,
    ) -> None:
        trainer = self.setup()

        rank = int(os.environ["RANK"])
        leader_rank = int(os.environ["CLIENT_LEADER_RANK"])

        model_path = Path(model_path)

        round_root = (
            Path(os.environ["CLIENT_CHECKPOINT_ROOT"])
            / f"round_{round_id}"
        )

        distributed_root = round_root / "global_dcp"
        dcp_step_path = distributed_root / "step-0"
        flat_model_path = round_root / "global_flat_model.pt"

        if rank == leader_rank:
            payload = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
            )

            flat_state = payload.get("model", payload)
            torch.save(flat_state, flat_model_path)

            torch_save_to_dcp(
                str(flat_model_path),
                str(dcp_step_path),
            )

            flat_model_path.unlink(missing_ok=True)

        dist.barrier()

        original_folder = trainer.checkpointer.folder
        trainer.checkpointer.folder = str(distributed_root)

        try:
            loaded = trainer.checkpointer.load(step=0)

            if not loaded:
                raise RuntimeError(
                    f"Failed to load global model from {dcp_step_path}"
                )
        finally:
            trainer.checkpointer.folder = original_folder

        # FedAvg changed the parameters. Reset client-local optimizer state.
        for optimizer in trainer.optimizers:
            optimizer.state.clear()

        dist.barrier()

    
    def close(self) -> None:
        self._trainer = None
        self._config = None
