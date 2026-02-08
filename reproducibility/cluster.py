"""Shared submitit wrapper for SLURM cluster job submission.

Provides YAML-based configuration and helpers for submitting
reproducibility script workloads to a SLURM cluster.

Usage:
    from cluster import SlurmConfig, submit_jobs, collect_results

    config = SlurmConfig.from_yaml("configs/slurm_default.yaml")
    jobs = submit_jobs(my_fn, args_list, config, log_dir="logs/my_script")
    # ... later, after jobs complete ...
    results = collect_results("logs/my_script")
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class SlurmConfig:
    """SLURM parameters loaded from a YAML config file."""

    partition: str = "cpu"
    timeout_min: int = 360
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 8
    gpus_per_node: int = 0
    mem_gb: int = 32
    setup: list[str] = field(default_factory=list)
    additional_parameters: dict[str, Any] = field(
        default_factory=dict
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SlurmConfig:
        """Load config from a YAML file."""
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        slurm_section = raw.get("slurm", raw)
        known = {
            f.name for f in dataclasses.fields(cls)
        }
        return cls(
            **{
                k: v
                for k, v in slurm_section.items()
                if k in known
            }
        )

    def to_submitit_params(self) -> dict[str, Any]:
        """Convert to kwargs for executor.update_parameters()."""
        params: dict[str, Any] = {
            "timeout_min": self.timeout_min,
            "slurm_partition": self.partition,
            "nodes": self.nodes,
            "slurm_ntasks_per_node": self.ntasks_per_node,
            "cpus_per_task": self.cpus_per_task,
            "gpus_per_node": self.gpus_per_node,
            "mem_gb": self.mem_gb,
        }
        if self.setup:
            params["slurm_setup"] = self.setup
        if self.additional_parameters:
            params[
                "slurm_additional_parameters"
            ] = self.additional_parameters
        return params


def make_executor(
    log_dir: str | Path,
    slurm_config: SlurmConfig | None = None,
    local: bool = False,
) -> Any:
    """Create a configured submitit executor.

    Args:
        log_dir: Directory for submitit logs.
        slurm_config: SLURM parameters. Uses defaults if None.
        local: Use DebugExecutor (runs in-process).
    """
    import submitit

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if local:
        return submitit.DebugExecutor(folder=str(log_dir))

    executor = submitit.AutoExecutor(folder=str(log_dir))
    if slurm_config is None:
        slurm_config = SlurmConfig()
    executor.update_parameters(
        **slurm_config.to_submitit_params()
    )
    return executor


def submit_jobs(
    fn: Callable[..., T],
    args_list: list[tuple[Any, ...]],
    slurm_config: SlurmConfig,
    log_dir: str | Path,
    local: bool = False,
) -> list[Any]:
    """Submit function calls as SLURM jobs.

    Args:
        fn: Callable to execute on each worker.
        args_list: List of argument tuples, one per job.
        slurm_config: SLURM parameters.
        log_dir: Directory for logs and job metadata.
        local: Run in-process for debugging.

    Returns:
        List of submitit Job objects.
    """
    executor = make_executor(
        log_dir=log_dir,
        slurm_config=slurm_config,
        local=local,
    )
    jobs = executor.map_array(fn, *zip(*args_list))

    print(f"Submitted {len(jobs)} jobs for {fn.__name__}")
    for job in jobs:
        print(f"  Job {job.job_id}")

    return jobs


def save_job_metadata(
    jobs: list[Any],
    args_list: list[tuple[Any, ...]],
    path: str | Path,
) -> None:
    """Save job IDs and arguments for later collection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "jobs": [
            {
                "job_id": str(job.job_id),
                "args": list(args),
            }
            for job, args in zip(jobs, args_list)
        ]
    }
    path.write_text(json.dumps(metadata, indent=2))
    print(f"Job metadata saved to {path}")


def collect_results(
    path: str | Path,
    log_dir: str | Path,
) -> list[dict[str, Any]]:
    """Collect results from previously submitted SLURM jobs.

    Args:
        path: Path to the job metadata JSON file.
        log_dir: The log directory used when submitting.

    Returns:
        List of result dicts from completed jobs.
    """
    import submitit

    path = Path(path)
    metadata = json.loads(path.read_text())

    results = []
    for entry in metadata["jobs"]:
        job_id = entry["job_id"]
        job = submitit.SlurmJob(
            folder=str(log_dir), job_id=job_id
        )
        try:
            result = job.result()
            results.append(result)
        except Exception as e:
            print(
                f"Job {job_id} (args={entry['args']}) "
                f"failed: {e}"
            )
    return results
