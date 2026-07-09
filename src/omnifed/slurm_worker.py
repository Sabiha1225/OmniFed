
# src/omnifed/slurm_worker.py
from __future__ import annotations
import argparse, json, os, signal, time, subprocess, pickle, warnings
from typing import Any, Dict, Optional
import numpy as np
from dataclasses import is_dataclass, asdict, dataclass
import csv

from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import torch.distributed as dist

from src.omnifed.communicator import AggregationOp
from src.omnifed.utils import print  # pretty printer used elsewhere

import threading
from datetime import datetime

from src.omnifed.trainer import create_backend
from pathlib import Path
from src.omnifed.communicator.grpc import GrpcCommunicator
import shutil


@dataclass(frozen=True)
class SlurmRole:
    role: str
    federated_rank: int
    client_id: int | None
    torchtitan_rank: int | None
    torchtitan_world_size: int | None
    is_client_leader: bool

# def resolve_subcluster_role(cfg) -> SlurmRole:
#     allocation_node_id = int(os.environ["SLURM_NODEID"])
#     local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

#     nodes_per_client = int(cfg.torchtitan.subclusters.nodes_per_client)
#     gpus_per_node = int(cfg.torchtitan.subclusters.gpus_per_node)
#     leader_rank = int(cfg.torchtitan.subclusters.leader_rank)

#     # First allocation node is automatically the Omnifed server.
#     if allocation_node_id == 0:
#         return SlurmRole(
#             role="server",
#             federated_rank=0,
#             client_id=None,
#             torchtitan_rank=None,
#             torchtitan_world_size=None,
#             is_client_leader=False,
#         )

#     client_node_index = allocation_node_id - 1
#     client_id = client_node_index // nodes_per_client
#     node_rank_inside_client = client_node_index % nodes_per_client

#     torchtitan_rank = node_rank_inside_client * gpus_per_node + local_rank
#     torchtitan_world_size = nodes_per_client * gpus_per_node

#     return SlurmRole(
#         role="client",
#         federated_rank=client_id + 1,
#         client_id=client_id,
#         torchtitan_rank=torchtitan_rank,
#         torchtitan_world_size=torchtitan_world_size,
#         is_client_leader=torchtitan_rank == leader_rank,
#     )


def resolve_subcluster_role(cfg) -> SlurmRole:
    role = os.environ["OMNIFED_ROLE"]

    if role == "server":
        return SlurmRole(
            role="server",
            federated_rank=0,
            client_id=None,
            torchtitan_rank=None,
            torchtitan_world_size=None,
            is_client_leader=False,
        )

    client_id = int(os.environ["CLIENT_ID"])

    # Automatically provided by Slurm for this client srun step.
    torchtitan_rank = int(os.environ["SLURM_PROCID"])
    torchtitan_world_size = int(os.environ["SLURM_NTASKS"])

    leader_rank = int(cfg.torchtitan.subclusters.leader_rank)

    return SlurmRole(
        role="client",
        federated_rank=client_id + 1,
        client_id=client_id,
        torchtitan_rank=torchtitan_rank,
        torchtitan_world_size=torchtitan_world_size,
        is_client_leader=torchtitan_rank == leader_rank,
    )

def create_federated_communicator(cfg, federated_rank: int, server_addr: str):
    communicator = GrpcCommunicator(
        rank=federated_rank,
        world_size=int(cfg.torchtitan.subclusters.num_clients) + 1,
        master_addr=server_addr,
        master_port=int(cfg.torchtitan.federated.server_port),
        max_send_message_length=128 * 1024 * 1024,
        max_receive_message_length=128 * 1024 * 1024,
        aggregation_timeout=float(
            cfg.torchtitan.federated.aggregation_timeout
        ),
        client_timeout=float(
            cfg.torchtitan.federated.aggregation_timeout
        ),
    )
    communicator.setup()
    return communicator

# This chunks by parameter. If a single parameter exceeds 
# the gRPC message limit, that tensor must additionally be sliced or the gRPC limit increased.
def iter_state_chunks(state_dict, chunk_size_mb: int):
    limit = chunk_size_mb * 1024 * 1024
    chunk = {}
    chunk_bytes = 0

    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu()

        if tensor.is_floating_point():
            tensor = tensor.float()
        
        tensor_bytes = tensor.numel() * tensor.element_size()

        if chunk and chunk_bytes + tensor_bytes > limit:
            yield chunk
            chunk = {}
            chunk_bytes = 0

        chunk[name] = tensor
        chunk_bytes += tensor_bytes

    if chunk:
        yield chunk

def get_initial_model_paths(cfg) -> tuple[Path, Path]:
    initial_path = (
        Path(cfg.torchtitan.subclusters.checkpoint_root)
        / f"job_{os.environ['SLURM_JOB_ID']}"
        / "initial"
        / "model.pt"
    )
    ready_path = initial_path.with_suffix(".ready")
    return initial_path, ready_path


def maybe_dist_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def wait_for_file(
    path: Path,
    timeout: float = 1800,
    interval: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout

    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {path}"
            )
        time.sleep(interval)


def _first_host_from_nodelist() -> str:
    out = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
        text=True,
    )
    return out.strip().splitlines()[0]


def install_preemption_handlers(on_checkpoint):
    def _h(signum, frame):
        try:
            on_checkpoint(reason=f"signal:{signum}")
        finally:
            time.sleep(2)
            os._exit(0)
    signal.signal(getattr(signal, "SIGUSR1"), _h)
    signal.signal(getattr(signal, "SIGTERM"), _h)


def _safe_len_train(dm) -> int:
    try:
        return len(dm.train) if dm.train is not None else 0
    except Exception:
        return 0


def _resolve_device_auto(rank: Optional[int]) -> torch.device:
    """
    Auto device resolver: GPU if available, otherwise CPU.
    Uses round-robin by local rank when multiple GPUs are present.
    """

    # Auto-assignment with GPU detection
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("Auto: CPU (no GPUs available)")
        return torch.device("cpu")

    # Round-robin GPU assignment
    effective_rank = rank if rank is not None else 0
    if rank is None:
        warnings.warn("No rank provided, defaulting to GPU 0")

    gpu_id = effective_rank % gpu_count
    device_str = f"cuda:{gpu_id}"
    print(f"Auto: {device_str} (rank {effective_rank}, {gpu_count} GPUs)")
    return torch.device(device_str)


def _gpu_probe(prefix: str = "GPU"):
    """Best-effort CUDA probe for logs (does not crash if NVML is unavailable)."""
    try:
        print(
            f"{prefix} probe: torch.cuda.is_available()={torch.cuda.is_available()} | "
            f"device_count={torch.cuda.device_count()} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}"
        )
        if torch.cuda.is_available():
            cur = torch.cuda.current_device()
            print(
                f"{prefix} probe: current_device={cur} name={torch.cuda.get_device_name(cur)}"
            )
    except Exception as e:
        print(f"{prefix} probe: (non-fatal) {e}")

def _to_jsonable(obj):
    # dataclasses → dict
    if is_dataclass(obj):
        obj = asdict(obj)

    # basic types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # torch tensors / numpy arrays
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
            # 0-dim tensors / numpy scalars
            try:
                return obj.item()
            except Exception:
                pass
    except Exception:
        pass

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # dict / list / tuple
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    # common torch meta
    if str(type(obj)).startswith("<class 'torch.") or str(type(obj)).startswith("<class 'numpy."):
        return str(obj)

    # fallback: __dict__ or string
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    return str(obj)


def start_gpu_memory_logger(rank: int, log_dir: str, interval_sec: float = 5.0):
    """
    Periodically log GPU memory usage for rank 0 only.
    Works on ROCm through torch.cuda APIs.
    """
    if rank != 0:
        return None

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "rank0_gpu_memory.log")
    stop_event = threading.Event()

    def _logger():
        header_needed = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        with open(log_path, "a", buffering=1) as f:
            if header_needed:
                f.write(
                    "timestamp,rank,device,allocated_mb,reserved_mb,"
                    "max_allocated_mb,max_reserved_mb\n"
                )

            while not stop_event.is_set():
                try:
                    if torch.cuda.is_available():
                        device_idx = torch.cuda.current_device()
                        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
                        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
                        max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)
                        max_reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2)

                        f.write(
                            f"{datetime.now().isoformat()},{rank},{device_idx},"
                            f"{allocated:.2f},{reserved:.2f},"
                            f"{max_allocated:.2f},{max_reserved:.2f}\n"
                        )
                    else:
                        f.write(f"{datetime.now().isoformat()},{rank},NA,NA,NA,NA,NA\n")
                except Exception as e:
                    f.write(f"{datetime.now().isoformat()},{rank},ERROR,{str(e)},,,\n")

                stop_event.wait(interval_sec)

    t = threading.Thread(target=_logger, daemon=True)
    t.start()
    return stop_event, log_path


def log_gpu_memory_snapshot(rank: int, log_path: str, tag: str):
    """
    Write one on-demand GPU memory snapshot for rank 0 only.
    """
    if rank != 0 or not torch.cuda.is_available():
        return

    try:
        device_idx = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2)

        with open(log_path, "a", buffering=1) as f:
            f.write(
                f"{datetime.now().isoformat()},{rank},{device_idx},"
                f"{allocated:.2f},{reserved:.2f},"
                f"{max_allocated:.2f},{max_reserved:.2f},tag={tag}\n"
            )
    except Exception:
        pass


def run_client_global_communication(
    cfg,
    communicator,
    result,
):
    payload = torch.load(
        result["checkpoint_path"],
        map_location="cpu",
    )

    local_state = payload["model"]
    local_tokens = float(payload["num_tokens"])

    total_tokens = communicator.aggregate(
        torch.tensor([local_tokens], dtype=torch.float64),
        AggregationOp.SUM,
    ).item()

    weight = local_tokens / max(total_tokens, 1.0)
    aggregated_state = {}

    chunk_size = int(cfg.torchtitan.federated.grpc_chunk_size_mb)

    for chunk in iter_state_chunks(local_state, chunk_size):
        weighted_chunk = {
            name: tensor * weight
            for name, tensor in chunk.items()
        }

        result_chunk = communicator.aggregate(
            weighted_chunk,
            AggregationOp.SUM,
        )
        aggregated_state.update(result_chunk)

    return aggregated_state

def run_torchtitan_client(cfg, role: SlurmRole) -> None:
    os.environ["RANK"] = str(role.torchtitan_rank)
    os.environ["WORLD_SIZE"] = str(role.torchtitan_world_size)
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    # os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = os.environ["CLIENT_MASTER_ADDR"]
    os.environ["MASTER_PORT"] = os.environ["CLIENT_MASTER_PORT"]
    os.environ["CLIENT_ID"] = str(role.client_id)
    os.environ["CLIENT_LEADER_RANK"] = str(
        cfg.torchtitan.subclusters.leader_rank
    )

    checkpoint_root = Path(os.environ["CLIENT_CHECKPOINT_ROOT"])

    backend = create_backend(
        cfg=cfg,
        backend_name="torchtitan",
    )

    communicator = None
    if role.is_client_leader:
        communicator = create_federated_communicator(
            cfg=cfg,
            federated_rank=role.federated_rank,
            server_addr=os.environ["SERVER_ADDR"],
        )

    initial_path, initial_ready_path = get_initial_model_paths(cfg)

    try:
        # Client 0 collectively creates the initial TorchTitan model.
        if role.client_id == 0:
            initial_result = backend.save_and_consolidate(
                round_id=-1,
                num_tokens=0,
            )

            if role.is_client_leader:
                initial_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                temporary_path = initial_path.with_suffix(".tmp")

                shutil.copyfile(
                    initial_result["checkpoint_path"],
                    temporary_path,
                )
                os.replace(temporary_path, initial_path)

                # Publish only after model.pt is completely written.
                initial_ready_path.touch()

        # Other clients wait for client 0 to publish the model.
        wait_for_file(initial_ready_path)

        # Every rank within this subcluster reaches this point.
        # dist.barrier()

        # Load initial ordinary tensors into this client's PP/TP model.
        backend.load_global_model(
            model_path=str(initial_path),
            round_id=-1,
        )

        # dist.barrier()
        maybe_dist_barrier()

        # Federated rounds.
        for round_id in range(int(cfg.global_rounds)):
            train_result = backend.train_local_steps(
                steps=int(cfg.torchtitan.federated.local_steps)
            )

            local_result = backend.save_and_consolidate(
                round_id=round_id,
                num_tokens=int(train_result["num_tokens"]),
            )

            global_path = (
                checkpoint_root
                / f"round_{round_id}"
                / "global_model.pt"
            )

            global_ready_path = global_path.with_suffix(".ready")

            if role.is_client_leader:
                assert communicator is not None

                aggregated_state = run_client_global_communication(
                    cfg=cfg,
                    communicator=communicator,
                    result=local_result,
                )

                global_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                temporary_path = global_path.with_suffix(".tmp")

                torch.save(
                    {"model": aggregated_state},
                    temporary_path,
                )

                # Atomic publish: model.pt appears only after the full file is written.
                os.replace(temporary_path, global_path)

                # Signal non-leader ranks that this round's global model is ready.
                global_ready_path.touch()

            # Non-leader ranks wait through Lustre, not raw dist.barrier().
            wait_for_file(global_ready_path)

            # Wait for this client's leader to finish writing.
            # dist.barrier()
            # maybe_dist_barrier()

            # Convert global tensors back into PP/TP DTensors.
            backend.load_global_model(
                model_path=str(global_path),
                round_id=round_id,
            )

            # dist.barrier()
            maybe_dist_barrier()

    finally:

        try:
            maybe_dist_barrier()
        except Exception:
            pass

        if communicator is not None:
            communicator.close()

        backend.close()

# def run_federated_server(cfg, role: SlurmRole) -> None:
#     assert role.federated_rank == 0

#     num_clients = int(cfg.torchtitan.subclusters.num_clients)

#     server = CheckpointAggregationServer(
#         num_clients=num_clients,
#         checkpoint_root=cfg.torchtitan.subclusters.checkpoint_root,
#         port=int(cfg.torchtitan.federated.server_port),
#     )

#     server.run(global_rounds=int(cfg.global_rounds))


def run_federated_server(cfg, role: SlurmRole) -> None:
    server_addr = subprocess.check_output(
        ["hostname"], text=True
    ).strip()

    communicator = create_federated_communicator(
        cfg=cfg,
        federated_rank=0,
        server_addr=server_addr,
    )

    initial_path, initial_ready_path = get_initial_model_paths(cfg)

    wait_for_file(initial_ready_path)

    initial = torch.load(
        initial_path,
        map_location="cpu",
        weights_only=False,
    )
    global_state = initial.get("model", initial)

    chunk_size = int(cfg.torchtitan.federated.grpc_chunk_size_mb)

    for round_id in range(int(cfg.global_rounds)):
        # Make the current global state available to client leaders.
        # communicator.broadcast(global_state)

        # Server participates with zero samples and zero-valued tensors.
        communicator.aggregate(
            torch.tensor([0.0], dtype=torch.float64),
            AggregationOp.SUM,
        )

        new_global_state = {}

        for chunk in iter_state_chunks(global_state, chunk_size):
            zero_chunk = {
                name: torch.zeros_like(tensor)
                for name, tensor in chunk.items()
            }

            aggregated_chunk = communicator.aggregate(
                zero_chunk,
                AggregationOp.SUM,
            )
            new_global_state.update(aggregated_chunk)

        global_state = new_global_state

        output = (
            Path(cfg.torchtitan.subclusters.checkpoint_root)
            / f"job_{os.environ['SLURM_JOB_ID']}"
            / "server"
            / f"round_{round_id}"
            / "model.pt"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": global_state}, output)

    communicator.close()


def main():
    # ---------- args & frozen config ----------
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-json", required=True, help="Path to engine_frozen.json written by Engine")
    args = p.parse_args()

    with open(args.cfg_json, "r") as f:
        raw = json.load(f)

    cfg = OmegaConf.create(raw["cfg"])          # EngineConfig-like dict (resolved)
    hydra_out_dir = raw["hydra_output_dir"]
    ckpt_dir = raw.get("slurm_checkpoint_dir") or os.path.join(hydra_out_dir, "engine", "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    subclusters = getattr(cfg.torchtitan, "subclusters", None)

    if subclusters is not None and subclusters.enabled:
        role = resolve_subcluster_role(cfg)

        if role.role == "server":
            run_federated_server(cfg, role)
        else:
            run_torchtitan_client(cfg, role)

        return

    # ---------- Slurm sizing & env ----------
    rank  = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

    # os.environ.setdefault("MASTER_ADDR", _first_host_from_nodelist())
    # os.environ.setdefault("MASTER_PORT", str(cfg.topology.local_comm.master_port))
    # os.environ.setdefault("RANK", str(rank))
    # os.environ.setdefault("WORLD_SIZE", str(world))

    master_addr = _first_host_from_nodelist()
    master_port = str(cfg.topology.local_comm.master_port)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)

    # Patch communicator config so it does not keep localhost/127.0.0.1
    if hasattr(cfg.topology, "local_comm") and cfg.topology.local_comm is not None:
        if "master_addr" in cfg.topology.local_comm:
            cfg.topology.local_comm["master_addr"] = master_addr
        if "master_port" in cfg.topology.local_comm:
            cfg.topology.local_comm["master_port"] = master_port

    if hasattr(cfg.topology, "global_comm") and cfg.topology.global_comm is not None:
        if "master_addr" in cfg.topology.global_comm:
            cfg.topology.global_comm["master_addr"] = master_addr
        if "master_port" in cfg.topology.global_comm:
            cfg.topology.global_comm["master_port"] = master_port

    print(
        f"[main] patched communicator config: "
        f"local_comm.master_addr={cfg.topology.local_comm.master_addr} "
        f"local_comm.master_port={cfg.topology.local_comm.master_port}",
        flush=True,
    )

    print(f"[main]  rank={rank} world={world} MASTER_ADDR={os.environ['MASTER_ADDR']} PORT={os.environ['MASTER_PORT']}", flush=True)
    _gpu_probe(prefix=f"[rank{rank}]")

    # ---------- Build topology & pick this node ----------
    topology = instantiate(cfg.topology, _recursive_=False)
    topology.setup(
        default_algorithm_cfg=cfg.algorithm,
        default_model_cfg=cfg.model,
        default_datamodule_cfg=cfg.datamodule,
    )
    node_configs = list(topology)
    if world != len(node_configs):
        print(f"[slurm_worker][FATAL] SLURM tasks ({world}) != topology nodes ({len(node_configs)})", flush=True)
        os._exit(1)
    node_cfg = node_configs[rank]

    # ---------- Paths (match Ray Node behavior) ----------
    node_name = node_cfg.name
    log_dir_base = getattr(node_cfg, "log_dir_base", None) or hydra_out_dir
    node_log_dir = os.path.join(log_dir_base, node_name)
    os.makedirs(node_log_dir, exist_ok=True)

    # ---------- Instantiate communicator/model/datamodule/algorithm ----------
    local_comm  = instantiate(node_cfg.local_comm)  # default _recursive_=True
    global_comm = instantiate(node_cfg.global_comm) if getattr(node_cfg, "global_comm", None) else None
    model       = instantiate(cfg.model)
    datamodule  = instantiate(cfg.datamodule)
    algorithm   = instantiate(node_cfg.algorithm, log_dir=node_log_dir)

    algorithm.use_torchtitan_backend = (
        os.environ.get("OMNIFED_INTERNAL_BACKEND", "torchdist") == "torchtitan"
    )
    if algorithm.use_torchtitan_backend:
        algorithm.torchtitan_backend = create_backend(
            cfg=cfg,
            backend_name="torchtitan",
            module=getattr(cfg, "torchtitan", None).module if getattr(cfg, "torchtitan", None) is not None else None,
            config_name=getattr(cfg, "torchtitan", None).config_name if getattr(cfg, "torchtitan", None) is not None else None,
            output_dir=getattr(cfg, "torchtitan", None).output_dir if getattr(cfg, "torchtitan", None) is not None else None,
            update_dir=getattr(cfg, "torchtitan", None).update_dir if getattr(cfg, "torchtitan", None) is not None else None,
        )

    # ---------- Device selection ----------
    # If communicator backend is NCCL, we must put tensors on CUDA before collectives.
    backend = getattr(local_comm, "backend", "gloo").lower()
    use_cuda = (backend == "nccl") and torch.cuda.is_available()
    device = _resolve_device_auto(local_rank) if use_cuda else torch.device("cpu")
    original_device = next(model.parameters()).device
    model = model.to(device, non_blocking=True)

    mem_logger = start_gpu_memory_logger(
        rank=rank,
        log_dir=os.path.join(hydra_out_dir, "engine"),
        interval_sec=10.0,
    )

    if mem_logger is not None:
        mem_stop_event, mem_log_path = mem_logger
        print(f"[main] rank 0 GPU memory log -> {mem_log_path}", flush=True)
        log_gpu_memory_snapshot(rank, mem_log_path, "after_model_to_device")
    else:
        mem_stop_event = None
        mem_log_path = ""

    # ---------- Init process group via communicator ----------
    local_comm.setup()
    if global_comm:
        global_comm.setup()
    print("[main]  communicator initialized.", flush=True)

    # ---------- Device selection ----------
    # If communicator backend is NCCL, we must put tensors on CUDA before collectives.
    # backend = getattr(local_comm, "backend", "gloo").lower()
    # use_cuda = (backend == "nccl") and torch.cuda.is_available()
    # device = _resolve_device_auto(local_rank) if use_cuda else torch.device("cpu")
    # original_device = next(model.parameters()).device
    # model = model.to(device, non_blocking=True)

    # backend = getattr(local_comm, "backend", "gloo")
    # device = _resolve_device_auto(local_rank if backend == "nccl" else None)
    # if backend == "nccl" and device.type == "cuda":
    #     torch.cuda.set_device(device.index or 0)
    #     model = model.to(device, non_blocking=True)

    # ---------- DataModule: setup (if it exists) ----------
    if hasattr(datamodule, "setup"):
        datamodule.setup()

    # ---------- Broadcast initial model (model is on CUDA when using NCCL) ----------
    if global_comm:
        model = global_comm.broadcast(model)
    model = local_comm.broadcast(model)

    # ---------- Compute group maxima (put tiny tensors on CUDA for NCCL) ----------
    #ctrl_dev = device if backend == "nccl" and device.type == "cuda" else torch.device("cpu")
    local_iters_per_epoch = _safe_len_train(datamodule)

    group_max = local_comm.aggregate(
        dict(
            iters_per_epoch=torch.tensor(local_iters_per_epoch, dtype=torch.int, device=device),
            epochs_per_round=torch.tensor(getattr(algorithm, "max_epochs_per_round", 1), dtype=torch.int, device=device),
        ),
        AggregationOp.MAX,
    )

    # Read them back on CPU safely
    iters_per_epoch  = int(group_max["iters_per_epoch"].detach().cpu().item())
    epochs_per_round = int(group_max["epochs_per_round"].detach().cpu().item())
    total_rounds     = int(cfg.global_rounds)

    # ---------- Hand off to algorithm.setup(...) exactly like Node ----------
    algorithm.setup(
        local_comm,
        global_comm,
        model,
        datamodule,
        iters_per_epoch,
        epochs_per_round,
        total_rounds,
    )

    # Ensure the algorithm's model lives on our chosen device
    # try:
    #     alg_dev = next(algorithm.local_model.parameters()).device
    # except Exception:
    #     alg_dev = torch.device("cpu")

    # if backend == "nccl" and device.type == "cuda" and alg_dev.type != "cuda":
    #     algorithm.local_model = algorithm.local_model.to(device, non_blocking=True)
    #     alg_dev = device

    algorithm.local_model = algorithm.local_model.to(device, non_blocking=True)

    print(f"[main]  training on device: {device}")

    # ---------- Train rounds like Node.run_experiment ----------
    try:
        for r in range(algorithm.max_rounds):
            algorithm.round_exec(r, algorithm.max_rounds)
    finally:
        log_gpu_memory_snapshot(rank, mem_log_path, "finally_before_restore")

        if mem_stop_event is not None:
            mem_stop_event.set()
        # Best-effort teardown
        # Restore original device placement
        algorithm.local_model = algorithm.local_model.to(original_device)
        print(
            f"Model restored to original device: {original_device}",
            flush=True,
        )
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    # ---------- Persist per-rank results ----------
    results = algorithm.get_experiment_data()
    node_results_dir = os.path.join(hydra_out_dir, "engine", "node_results")
    os.makedirs(node_results_dir, exist_ok=True)
    out_path = os.path.join(node_results_dir, f"node_{rank:03d}_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    out_path_json = os.path.join(node_results_dir, f"node_{rank:03d}_results.json")
    with open(out_path_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)
    print(f"[slurm_worker] wrote JSON results -> {out_path_json}", flush=True)

    print(f"[main]  wrote results -> {out_path}", flush=True)
    print(f"[main]  rank={rank} finished.", flush=True)


if __name__ == "__main__":
    main()
