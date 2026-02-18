# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import gc
import getpass
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypeVar

import questionary
import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import DatasetDict, ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from optuna import Trial
from psutil import Process
from questionary import Choice, Style
from rich.console import Console

from .config import DatasetSpecification, Settings

print = Console(highlight=False).print


# Global cache for device information to avoid repeated queries
_device_info_cache: dict[str, Any] = {}


def get_cached_device_info(force_refresh: bool = False) -> dict[str, Any]:
    """
    Get cached device information to reduce overhead of repeated queries.
    
    Args:
        force_refresh: If True, bypass cache and query devices again.
    
    Returns:
        Dictionary with device counts and capabilities.
    """
    global _device_info_cache
    
    if not force_refresh and _device_info_cache:
        return _device_info_cache
    
    info = {
        "has_cuda": torch.cuda.is_available(),
        "has_xpu": is_xpu_available(),
        "has_mps": torch.backends.mps.is_available(),
        "has_mlu": is_mlu_available(),
        "has_musa": is_musa_available(),
        "has_sdaa": is_sdaa_available(),
    }
    
    if info["has_cuda"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(i) for i in range(info["cuda_device_count"])
        ]
    elif info["has_xpu"]:
        info["xpu_device_count"] = torch.xpu.device_count()
        info["xpu_device_names"] = [
            torch.xpu.get_device_name(i) for i in range(info["xpu_device_count"])
        ]
    
    _device_info_cache = info
    return info


def prewarm_gpu_memory(device_count: int, prewarm_size_mb: int = 100) -> bool:
    """
    Pre-allocate and free memory on GPUs to reduce fragmentation and prevent OOM.
    
    Args:
        device_count: Number of GPU devices to prewarm.
        prewarm_size_mb: Size of prewarm allocation in MB per device.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        if not torch.cuda.is_available():
            return False
        
        warmup_tensors = []
        size = prewarm_size_mb * 1024 * 1024 // 4  # 4 bytes per float32
        
        for i in range(device_count):
            # Allocate a tensor on each device
            tensor = torch.zeros(size, dtype=torch.float32, device=f"cuda:{i}")
            warmup_tensors.append(tensor)
        
        # Free the tensors
        del warmup_tensors
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"[yellow]Warning: GPU memory pre-warming failed: {e}[/]")
        return False


def print_memory_usage():
    def p(label: str, size_in_bytes: int):
        print(f"[grey50]{label}: [bold]{size_in_bytes / (1024**3):.2f} GB[/][/]")

    p("Resident system RAM", Process().memory_info().rss)

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        allocated = sum(torch.cuda.memory_allocated(device) for device in range(count))
        reserved = sum(torch.cuda.memory_reserved(device) for device in range(count))
        p("Allocated GPU VRAM", allocated)
        p("Reserved GPU VRAM", reserved)
    elif is_xpu_available():
        count = torch.xpu.device_count()
        allocated = sum(torch.xpu.memory_allocated(device) for device in range(count))
        reserved = sum(torch.xpu.memory_reserved(device) for device in range(count))
        p("Allocated XPU memory", allocated)
        p("Reserved XPU memory", reserved)
    elif torch.backends.mps.is_available():
        p("Allocated MPS memory", torch.mps.current_allocated_memory())
        p("Driver (reserved) MPS memory", torch.mps.driver_allocated_memory())


def get_device_memory_info() -> dict[int, dict[str, int]]:
    """
    Get detailed memory information for each GPU device.
    
    Returns:
        Dictionary mapping device index to memory info (total, free, allocated, reserved).
    """
    device_info = {}
    
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            free, total = torch.cuda.mem_get_info(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            device_info[i] = {
                "total": total,
                "free": free,
                "allocated": allocated,
                "reserved": reserved,
            }
    elif is_xpu_available():
        count = torch.xpu.device_count()
        for i in range(count):
            allocated = torch.xpu.memory_allocated(i)
            reserved = torch.xpu.memory_reserved(i)
            device_info[i] = {
                "total": 0,  # XPU doesn't provide total memory info
                "free": 0,
                "allocated": allocated,
                "reserved": reserved,
            }
    
    return device_info


def get_auto_max_memory(reserve_ratio: float = 0.15) -> dict[str, str]:
    """
    Automatically determine max_memory settings for multi-GPU setups.
    
    Args:
        reserve_ratio: Fraction of memory to reserve (0.0-1.0). Default 0.15 (15%).
    
    Returns:
        Dictionary with per-device memory limits suitable for accelerate's max_memory parameter.
    """
    max_memory = {}
    
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            free, total = torch.cuda.mem_get_info(i)
            # Use free memory minus reserve to avoid OOM
            usable = int(free * (1.0 - reserve_ratio))
            # Convert to GB, but use MB if less than 1GB for better granularity
            if usable >= 1024**3:
                max_memory[str(i)] = f"{usable // (1024**3)}GB"
            else:
                max_memory[str(i)] = f"{usable // (1024**2)}MB"
    elif is_xpu_available():
        # XPU doesn't provide memory info, so we can't auto-configure
        pass
    
    return max_memory


def check_device_health() -> bool:
    """
    Perform health checks on available GPU devices.
    
    Returns:
        True if all devices are healthy, False otherwise.
    """
    all_healthy = True
    
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            try:
                # Test basic memory operations
                free, total = torch.cuda.mem_get_info(i)
                
                # Check if device has reasonable free memory (at least 100MB)
                if free < 100 * (1024**2):
                    print(
                        f"[yellow]Warning: GPU {i} has very little free memory ({free / (1024**2):.0f} MB)[/]"
                    )
                    all_healthy = False
                
                # Test tensor allocation on device
                test_tensor = torch.zeros(1, device=f"cuda:{i}")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[red]Error: GPU {i} health check failed: {e}[/]")
                all_healthy = False
    
    return all_healthy


def print_per_device_memory_usage():
    """
    Print detailed memory usage for each GPU device.
    """
    device_info = get_device_memory_info()
    
    if not device_info:
        return
    
    print()
    print("[bold]Per-device memory usage:[/]")
    
    for device_id, info in device_info.items():
        if info["total"] > 0:
            used_pct = (info["allocated"] / info["total"]) * 100
            print(
                f"  GPU {device_id}: "
                f"[bold]{info['allocated'] / (1024**3):.2f} GB[/] / "
                f"{info['total'] / (1024**3):.2f} GB "
                f"([bold]{used_pct:.1f}%[/] used)"
            )
        else:
            print(
                f"  Device {device_id}: "
                f"[bold]{info['allocated'] / (1024**3):.2f} GB[/] allocated"
            )


def is_notebook() -> bool:
    # Check for specific environment variables (Colab, Kaggle).
    # This is necessary because when running as a subprocess (e.g. !heretic),
    # get_ipython() might not be available or might not reflect the notebook environment.
    if os.getenv("COLAB_GPU") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return True

    # Check IPython shell type (for library usage).
    try:
        from IPython import get_ipython  # ty:ignore[unresolved-import]

        shell = get_ipython()
        if shell is None:
            return False

        shell_name = shell.__class__.__name__
        if shell_name in ["ZMQInteractiveShell", "Shell"]:
            return True

        if "google.colab" in str(shell.__class__):
            return True

        return False
    except (ImportError, NameError, AttributeError):
        return False


def prompt_select(message: str, choices: list[Any]) -> Any:
    if is_notebook():
        print()
        print(message)
        real_choices = []

        for i, choice in enumerate(choices, 1):
            if isinstance(choice, Choice):
                print(f"[{i}] {choice.title}")
                real_choices.append(choice.value)
            else:
                print(f"[{i}] {choice}")
                real_choices.append(choice)

        while True:
            try:
                selection = input("Enter number: ")
                index = int(selection) - 1
                if 0 <= index < len(real_choices):
                    return real_choices[index]
                print(
                    f"[red]Please enter a number between 1 and {len(real_choices)}[/]"
                )
            except ValueError:
                print("[red]Invalid input. Please enter a number.[/]")
    else:
        return questionary.select(
            message,
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        ).ask()


def prompt_text(
    message: str,
    default: str = "",
    qmark: str = "?",
    unsafe: bool = False,
) -> str:
    if is_notebook():
        print()
        result = input(f"{message} [{default}]: " if default else f"{message}: ")
        return result if result else default
    else:
        question = questionary.text(message, default=default, qmark=qmark)
        if unsafe:
            return question.unsafe_ask()
        else:
            return question.ask()


def prompt_path(message: str) -> str:
    if is_notebook():
        return prompt_text(message)
    else:
        return questionary.path(message, only_directories=True).ask()


def prompt_password(message: str) -> str:
    if is_notebook():
        print()
        return getpass.getpass(message)
    else:
        return questionary.password(message).ask()


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


@dataclass
class Prompt:
    system: str
    user: str


def load_prompts(
    settings: Settings,
    specification: DatasetSpecification,
) -> list[Prompt]:
    path = specification.dataset
    split_str = specification.split

    if os.path.isdir(path):
        if Path(path, DATASET_STATE_JSON_FILENAME).exists():
            # Dataset saved with datasets.save_to_disk; needs special handling.
            # Path should be the subdirectory for a particular split.
            dataset = load_from_disk(path)
            assert not isinstance(dataset, DatasetDict), (
                "Loading dataset dicts is not supported"
            )
            # Parse the split instructions.
            instruction = ReadInstruction.from_spec(split_str)
            # Associate the split with its number of examples (lines).
            split_name = str(dataset.split)
            name2len = {split_name: len(dataset)}
            # Convert the instructions to absolute indices and select the first one.
            abs_instruction = instruction.to_absolute(name2len)[0]
            # Get the dataset by applying the indices.
            dataset = dataset[abs_instruction.from_ : abs_instruction.to]
        else:
            # Path is a local directory.
            dataset = load_dataset(
                path,
                split=split_str,
                # Don't require the number of examples (lines) per split to be pre-defined.
                verification_mode=VerificationMode.NO_CHECKS,
                # But also don't use cached data, as the dataset may have changed on disk.
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
    else:
        # Probably a repository path; let load_dataset figure it out.
        dataset = load_dataset(path, split=split_str)

    prompts = list(dataset[specification.column])

    if specification.prefix:
        prompts = [f"{specification.prefix} {prompt}" for prompt in prompts]

    if specification.suffix:
        prompts = [f"{prompt} {specification.suffix}" for prompt in prompts]

    system_prompt = (
        settings.system_prompt
        if specification.system_prompt is None
        else specification.system_prompt
    )

    return [
        Prompt(
            system=system_prompt,
            user=prompt,
        )
        for prompt in prompts
    ]


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def empty_cache():
    # Collecting garbage is not an idempotent operation, and to avoid OOM errors,
    # gc.collect() has to be called both before and after emptying the backend cache.
    # See https://github.com/p-e-w/heretic/pull/17 for details.
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        torch.sdaa.empty_cache()  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        torch.musa.empty_cache()  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()


def get_trial_parameters(trial: Trial) -> dict[str, str]:
    params = {}

    direction_index = trial.user_attrs["direction_index"]
    params["direction_index"] = (
        "per layer" if (direction_index is None) else f"{direction_index:.2f}"
    )

    for component, parameters in trial.user_attrs["parameters"].items():
        for name, value in parameters.items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[Prompt],
) -> str:
    if Path(settings.model).exists():
        # Hide the path, which may contain private information.
        model_link = "a model"
    else:
        model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic-llm")}

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.4f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""
