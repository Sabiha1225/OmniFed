from __future__ import annotations

import base64
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


def _inside_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


@dataclass
class SlurmConfig:
    enabled: bool = False
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    time: str = "02:00:00"
    nodes: int = 2
    ntasks_per_node: int = 1
    cpus_per_task: int = 8

    # GPU options
    gres: Optional[str] = None
    gpus_per_node: int = 0
    gpus_per_task: Optional[int] = None
    gpu_bind: str = "closest"

    job_name: str = "omnifed"
    constraint: Optional[str] = None
    reservation: Optional[str] = None
    setup_lines: List[str] = field(default_factory=list)

    checkpoint_dir: Optional[str] = None
    preempt_signal: str = "USR1"
    preempt_notice_sec: int = 180
    resume_from: Optional[str] = None

    # launcher will set this so Slurm world size == topology size
    ntasks: Optional[int] = None

    # runtime fields (set by Engine)
    work_dir: Optional[str] = None
    cfg_json_path: Optional[str] = None
    pyexe: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    def sbatch_lines(self) -> List[str]:
        lines = [
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --signal=B:{self.preempt_signal}@{self.preempt_notice_sec}",
            f"#SBATCH -C nvme",
            # f"#SBATCH --exclusive",
        ]

        if self.ntasks:
            lines.append(f"#SBATCH --ntasks={self.ntasks}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.reservation:
            lines.append(f"#SBATCH --reservation={self.reservation}")
        if self.stdout:
            lines.append(f"#SBATCH -o {self.stdout}")
        if self.stderr:
            lines.append(f"#SBATCH -e {self.stderr}")
        if self.work_dir:
            lines.append(f"#SBATCH --chdir={self.work_dir}")

        # GPU request section
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        elif self.gpus_per_node and self.gpus_per_node > 0:
            lines.append(f"#SBATCH --gpus-per-node={int(self.gpus_per_node)}")

        if self.gpus_per_task is not None:
            lines.append(f"#SBATCH --gpus-per-task={int(self.gpus_per_task)}")

        return lines


class SlurmOnlyLauncher:
    """
    Submit sbatch from outside Slurm; inside allocation the worker module drives the run.
    """

    @staticmethod
    def submit_or_exit(sconf: SlurmConfig) -> None:
        assert not _inside_slurm(), "submit_or_exit() must be called outside Slurm."
        assert sconf.work_dir and sconf.cfg_json_path, "work_dir and cfg_json_path must be set"

        pyexe = sconf.pyexe or os.getenv("PYEXE") or "python"

        # Read JSON and base64-encode so we can broadcast to all nodes without scp
        with open(sconf.cfg_json_path, "rb") as f:
            payload_b64 = base64.b64encode(f.read()).decode("ascii")

        cfg_dir = os.path.dirname(sconf.cfg_json_path)

        # Compute project root (directory that contains "src")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        lines: List[str] = ["#!/bin/bash"]
        lines += sconf.sbatch_lines()
        lines += [
            "set -euo pipefail",
            f'export PYTHONPATH="${{PYTHONPATH:-}}:{repo_root}"',
            'export PYTHONUNBUFFERED=1',
            'export HYDRA_FULL_ERROR=1',
            'export OMNIFED_DEBUG=${OMNIFED_DEBUG:-0}',
            "",
        ]

        if sconf.setup_lines:
            lines += sconf.setup_lines + [""]

        lines += [
            'echo "SLURM_JOB_ID=$SLURM_JOB_ID"',
            'echo "SLURM_NODELIST=$SLURM_NODELIST"',
            'echo "Running on $(hostname)"',
            'echo "PYTHONPATH=$PYTHONPATH"',
            f'echo "Requested PYEXE={pyexe}"',
            "",
            # Ensure the JSON directory exists on every node
            f'srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc {shlex.quote(f"mkdir -p {cfg_dir}")}',
            "",
            # Write identical JSON file on each node by decoding embedded base64
            f'export OMNIFED_CFG_B64="{payload_b64}"',
            'srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc ' +
            shlex.quote(f'echo "$OMNIFED_CFG_B64" | base64 -d > {sconf.cfg_json_path}'),
            'srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc ' +
            shlex.quote(
                f'echo "[$(hostname)] wrote {sconf.cfg_json_path}; size=$(stat -c%s {sconf.cfg_json_path}) bytes"'
            ),
            "",
            # Frontier / AMD fix for Ray import path
            "set -x",
            "srun --export=ALL bash -lc " + shlex.quote(
                'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then '
                'export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; '
                'fi; '
                'unset ROCR_VISIBLE_DEVICES; '
                'echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"; '
                'echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"; '
                'echo "Using worker python: ${PYEXE:-' + pyexe + '}"; '
                '${PYEXE:-' + pyexe + '} -u -m src.omnifed.slurm_worker '
                '--cfg-json ' + shlex.quote(sconf.cfg_json_path)
            ),
            "set +x",
        ]

        script = "\n".join(lines) + "\n"

        path = os.path.join(sconf.work_dir, "omnifed_slurm_only.sh")
        with open(path, "w") as f:
            f.write(script)
        os.chmod(path, 0o755)

        print("\n===== Generated sbatch (Slurm-only) =====\n")
        print(script)
        print("===== end sbatch =====\n")

        out = subprocess.check_output(["sbatch", path], text=True).strip()
        print(f"[SlurmOnlyLauncher] sbatch response: {out}")
        raise SystemExit(0)


class SlurmTorchTitanLauncher:
    @staticmethod
    def submit_or_exit(sconf: SlurmConfig, subcluster_cfg: dict) -> None:
        assert sconf.work_dir and sconf.cfg_json_path

        num_clients = int(subcluster_cfg["num_clients"])
        nodes_per_client = int(subcluster_cfg["nodes_per_client"])
        gpus_per_node = int(subcluster_cfg["gpus_per_node"])
        checkpoint_root = subcluster_cfg["checkpoint_root"]
        port_base = int(subcluster_cfg["master_port_base"])

        pyexe = sconf.pyexe or "python"
        leader_rank = int(subcluster_cfg["leader_rank"])

        #worker_entrypoint = (
        #    "bash -lc '"
        #    'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then '
        #    'export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; '
        #    'unset ROCR_VISIBLE_DEVICES; '
        #    'fi; '
        #    'echo "[worker] hostname=$(hostname) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"; '
        #    'exec "$PYEXE" -u -m src.omnifed.slurm_worker --cfg-json "$CFG_JSON"'
        #    "'"
        #)

        worker_entrypoint = (
            "bash -lc "
            + shlex.quote(
                'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then '
                'export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; '
                'fi; '
                'unset ROCR_VISIBLE_DEVICES; '
                'export HF_HOME="/mnt/bb/sabiha/hf_cache/${SLURM_JOB_ID}/rank_${SLURM_PROCID}"; '
                'export HF_DATASETS_CACHE="${HF_HOME}/datasets"; '
                'export TRANSFORMERS_CACHE="${HF_HOME}/transformers"; '
                'mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"; '
                'echo "[worker] hostname=$(hostname) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"; '
                'echo "HF_HOME=$HF_HOME"; '
                'echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"; '
                'exec "$PYEXE" -u -m src.omnifed.slurm_worker --cfg-json "$CFG_JSON"'
            )
        )

        lines = ["#!/bin/bash"]
        lines += sconf.sbatch_lines()
        lines += ["set -euo pipefail"]

        if sconf.setup_lines:
            lines += sconf.setup_lines + [""]

        lines += [
            f'export PYTHONPATH="{sconf.work_dir}:${{PYTHONPATH:-}}"',
            f'export PYEXE="${{PYEXE:-{pyexe}}}"',
            'mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")',
            'SERVER_HOST="${HOSTS[0]}"',
            f'export CFG_JSON="{sconf.cfg_json_path}"',
            f'CHECKPOINT_ROOT="{checkpoint_root}/job_${{SLURM_JOB_ID}}"',
            'mkdir -p "$CHECKPOINT_ROOT"',
            "",
            'srun --exclusive --nodes=1 --ntasks=1 '
            '--nodelist="$SERVER_HOST" '
            'env OMNIFED_ROLE=server '
            'FEDERATED_RANK=0 '
            'CHECKPOINT_ROOT="$CHECKPOINT_ROOT" '
            f'{worker_entrypoint} &',
        ]

        for client_id in range(num_clients):
            first_node = 1 + client_id * nodes_per_client
            world_size = nodes_per_client * gpus_per_node

            lines += [
                f'CLIENT_{client_id}_NODES=$(IFS=,; echo "${{HOSTS[*]:{first_node}:{nodes_per_client}}}")',
                f'CLIENT_{client_id}_MASTER="${{HOSTS[{first_node}]}}"',
                (
                    f"srun --exclusive --nodes={nodes_per_client} "
                    f"--ntasks={world_size} --ntasks-per-node={gpus_per_node} "
                    #"--gpus-per-task=1 --gpu-bind=closest "
                    f'--nodelist="$CLIENT_{client_id}_NODES" '
                    f'env OMNIFED_ROLE=client '
                    f'CLIENT_ID={client_id} '
                    f'FEDERATED_RANK={client_id + 1} '
                    f'CLIENT_LEADER_RANK={leader_rank} '
                    f'CLIENT_MASTER_ADDR="$CLIENT_{client_id}_MASTER" '
                    f'CLIENT_MASTER_PORT={port_base + client_id} '
                    f'SERVER_ADDR="$SERVER_HOST" '
                    f'CLIENT_CHECKPOINT_ROOT="$CHECKPOINT_ROOT/client_{client_id}" '
                    f'{worker_entrypoint} &'
                ),
            ]

        lines += [
            "wait",
            'echo "All Torchtitan subclusters completed"',
        ]

        script_path = os.path.join(
            sconf.work_dir, "omnifed_torchtitan_slurm.sh"
        )

        with open(script_path, "w") as file:
            file.write("\n".join(lines) + "\n")

        os.chmod(script_path, 0o755)
        response = subprocess.check_output(
            ["sbatch", script_path], text=True
        ).strip()
        print(response)
        raise SystemExit(0)
