from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SlurmConfig:
    enabled: bool = False
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    time: str = "02:00:00"

    nodes: int = 2
    ntasks_per_node: int = 1
    cpus_per_task: int = 6

    gres: Optional[str] = None
    gpus_per_node: int = 0
    gpus_per_task: Optional[int] = None
    gpu_bind: str = "closest"

    job_name: str = "omnifed"
    constraint: Optional[str] = None
    reservation: Optional[str] = None

    setup_lines: list[str] = field(
        default_factory=list
    )

    checkpoint_dir: Optional[str] = None
    preempt_signal: str = "USR1"
    preempt_notice_sec: int = 180
    resume_from: Optional[str] = None

    ntasks: Optional[int] = None

    work_dir: Optional[str] = None
    cfg_json_path: Optional[str] = None
    pyexe: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    def sbatch_lines(self) -> list[str]:
        lines = [
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --nodes={self.nodes}",
            (
                "#SBATCH --ntasks-per-node="
                f"{self.ntasks_per_node}"
            ),
            (
                "#SBATCH --cpus-per-task="
                f"{self.cpus_per_task}"
            ),
            f"#SBATCH --time={self.time}",
            (
                "#SBATCH --signal=B:"
                f"{self.preempt_signal}"
                f"@{self.preempt_notice_sec}"
            ),
            "#SBATCH -C nvme",
            "#SBATCH --exclusive",
        ]

        if self.ntasks is not None:
            lines.append(
                f"#SBATCH --ntasks={self.ntasks}"
            )

        if self.account:
            lines.append(
                f"#SBATCH --account={self.account}"
            )

        if self.partition:
            lines.append(
                f"#SBATCH --partition={self.partition}"
            )

        if self.qos:
            lines.append(
                f"#SBATCH --qos={self.qos}"
            )

        if self.constraint:
            lines.append(
                f"#SBATCH --constraint={self.constraint}"
            )

        if self.reservation:
            lines.append(
                f"#SBATCH --reservation={self.reservation}"
            )

        if self.stdout:
            lines.append(
                f"#SBATCH -o {self.stdout}"
            )

        if self.stderr:
            lines.append(
                f"#SBATCH -e {self.stderr}"
            )

        if self.work_dir:
            lines.append(
                f"#SBATCH --chdir={self.work_dir}"
            )

        if self.gres:
            lines.append(
                f"#SBATCH --gres={self.gres}"
            )
        elif self.gpus_per_node > 0:
            lines.append(
                "#SBATCH --gpus-per-node="
                f"{self.gpus_per_node}"
            )

        if self.gpus_per_task is not None:
            lines.append(
                "#SBATCH --gpus-per-task="
                f"{self.gpus_per_task}"
            )

        return lines