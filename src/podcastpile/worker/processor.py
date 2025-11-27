#!/usr/bin/env python3
"""
Worker processor for Podcast Pile - diarizes audio and uploads results
"""

import hashlib
import json
import logging
import multiprocessing
import os
import tempfile
import threading
import time
import uuid
import datetime
import gc
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio
from nemo.collections.asr.models import SortformerEncLabelModel
from tqdm import tqdm
from podcastpile.nisqa import NISQAPredictor

logger = logging.getLogger(__name__)

# Worker version - increment when making changes to processing logic
WORKER_VERSION = "0.5.0"  # Subprocess-based adaptive concurrency (safe multi-job per GPU)


def get_gpu_info(gpu_id: Optional[int] = None) -> Optional[str]:
    """Get GPU device information"""
    try:
        import torch

        if torch.cuda.is_available():
            device_id = gpu_id if gpu_id is not None else torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_id)
            return f"{gpu_name} (CUDA {device_id})"
    except Exception as e:
        logger.debug(f"Could not get GPU info: {e}")
    return None


def get_available_gpus() -> list:
    """Get list of available GPU IDs"""
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    return []


@dataclass
class GPUMemoryStats:
    """GPU memory statistics"""
    total_mb: float
    used_mb: float
    free_mb: float
    utilization_pct: float

    @property
    def available_mb(self) -> float:
        """Memory available for new work (conservative estimate)"""
        return self.free_mb

    @property
    def is_low(self) -> bool:
        """Check if memory is critically low (<15% free)"""
        return self.utilization_pct > 85.0

    @property
    def is_very_low(self) -> bool:
        """Check if memory is dangerously low (<10% free)"""
        return self.utilization_pct > 90.0


class GPUMemoryMonitor:
    """
    Monitors GPU memory usage and provides safe concurrency recommendations.

    The monitor tracks memory usage history and uses conservative estimates
    to prevent OOM errors while maximizing GPU utilization.
    """

    # Safety buffer: always keep this much memory free (MB)
    SAFETY_BUFFER_MB = 1024  # 1GB safety buffer

    # Minimum free memory percentage to allow new jobs
    MIN_FREE_PCT = 15.0

    # Memory usage history for smoothing
    HISTORY_SIZE = 10

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self._history: deque = deque(maxlen=self.HISTORY_SIZE)
        self._lock = threading.Lock()
        self._peak_usage_mb = 0.0
        self._job_memory_estimates: deque = deque(maxlen=50)  # Recent job memory usage

    def get_memory_stats(self) -> Optional[GPUMemoryStats]:
        """Get current GPU memory statistics"""
        try:
            # Use nvidia-smi for more accurate readings (includes all processes)
            total = torch.cuda.get_device_properties(self.gpu_id).total_memory
            # Get memory allocated by this process
            allocated = torch.cuda.memory_allocated(self.gpu_id)
            # Get memory reserved by this process (includes cached allocations)
            reserved = torch.cuda.memory_reserved(self.gpu_id)

            # For a more accurate "used" reading, we use nvidia-smi via pynvml
            # But fall back to reserved if pynvml isn't available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used = info.used
                free = info.free
            except ImportError:
                # Fallback: use torch's reserved memory as estimate
                # This is less accurate but still useful
                used = reserved
                free = total - reserved

            total_mb = total / (1024 * 1024)
            used_mb = used / (1024 * 1024)
            free_mb = free / (1024 * 1024)
            utilization_pct = (used_mb / total_mb) * 100 if total_mb > 0 else 0

            stats = GPUMemoryStats(
                total_mb=total_mb,
                used_mb=used_mb,
                free_mb=free_mb,
                utilization_pct=utilization_pct
            )

            with self._lock:
                self._history.append(stats)
                self._peak_usage_mb = max(self._peak_usage_mb, used_mb)

            return stats

        except Exception as e:
            logger.warning(f"Failed to get GPU memory stats: {e}")
            return None

    def record_job_memory_usage(self, memory_mb: float):
        """Record memory usage from a completed job for estimation"""
        with self._lock:
            self._job_memory_estimates.append(memory_mb)

    def get_estimated_job_memory_mb(self) -> float:
        """
        Estimate memory needed for a new job based on history.
        Uses conservative estimate (75th percentile of recent jobs).
        """
        with self._lock:
            if not self._job_memory_estimates:
                # Default estimate if no history: 4GB per job (conservative)
                return 4096.0

            estimates = list(self._job_memory_estimates)

        # Use 75th percentile for conservative estimate
        estimates.sort()
        idx = int(len(estimates) * 0.75)
        return estimates[min(idx, len(estimates) - 1)]

    def get_safe_concurrent_jobs(self, current_jobs: int = 0) -> int:
        """
        Calculate safe number of concurrent jobs based on available memory.

        Returns the recommended total number of concurrent jobs (not additional jobs).
        """
        stats = self.get_memory_stats()
        if not stats:
            # Can't determine memory, play it safe
            return max(1, current_jobs)

        estimated_job_memory = self.get_estimated_job_memory_mb()

        # Available memory for new jobs (with safety buffer)
        available_mb = stats.free_mb - self.SAFETY_BUFFER_MB

        if available_mb <= 0:
            # Memory is too low, don't start new jobs
            return current_jobs

        # Calculate how many additional jobs we can fit
        additional_jobs = int(available_mb / estimated_job_memory)

        # Cap at reasonable maximum (diminishing returns beyond ~4 concurrent jobs)
        max_concurrent = 4

        recommended = min(current_jobs + additional_jobs, max_concurrent)

        # Never recommend less than 1 if we have any free memory
        return max(1, recommended)

    def should_throttle(self) -> bool:
        """
        Check if we should throttle (pause new jobs) due to memory pressure.

        Returns True if memory is critically low.
        """
        stats = self.get_memory_stats()
        if not stats:
            return False  # Can't determine, don't throttle

        return stats.is_very_low

    def can_start_new_job(self) -> bool:
        """
        Check if it's safe to start a new job.

        More conservative than should_throttle - used before job acquisition.
        """
        stats = self.get_memory_stats()
        if not stats:
            return True  # Can't determine, allow it

        # Need at least estimated job memory + safety buffer free
        estimated_job_memory = self.get_estimated_job_memory_mb()
        required_free = estimated_job_memory + self.SAFETY_BUFFER_MB

        return stats.free_mb >= required_free

    def force_memory_cleanup(self):
        """Force GPU memory cleanup via garbage collection and cache clearing"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.gpu_id)
        logger.info(f"GPU {self.gpu_id}: Forced memory cleanup")


@dataclass
class AdaptiveJobSlot:
    """Represents a job slot with memory tracking"""
    job_id: int
    future: Optional[Future] = None
    start_time: float = field(default_factory=time.time)
    memory_before_mb: float = 0.0
    memory_peak_mb: float = 0.0


class AdaptiveJobScheduler:
    """
    Intelligently schedules concurrent jobs to maximize GPU utilization
    while preventing OOM errors.

    Features:
    - Starts conservative (1 job) and ramps up based on observed memory usage
    - Continuously monitors GPU memory and adjusts concurrency
    - Backs off immediately when memory gets tight
    - Learns from job memory patterns over time
    - Gracefully handles OOM by reducing concurrency
    """

    # Initial number of concurrent jobs (conservative start)
    INITIAL_CONCURRENCY = 1

    # How often to check memory and adjust (seconds)
    MEMORY_CHECK_INTERVAL = 2.0

    # Cooldown after reducing concurrency (seconds)
    BACKOFF_COOLDOWN = 30.0

    # Minimum time between concurrency increases (seconds)
    RAMP_UP_INTERVAL = 60.0

    def __init__(
        self,
        gpu_id: int,
        job_processor: Callable,
        max_concurrency: int = 4
    ):
        """
        Initialize the adaptive scheduler.

        Args:
            gpu_id: GPU device ID to manage
            job_processor: Callable that processes a single job (takes job dict, returns bool)
            max_concurrency: Maximum number of concurrent jobs (default: 4)
        """
        self.gpu_id = gpu_id
        self.job_processor = job_processor
        self.max_concurrency = max_concurrency

        self.memory_monitor = GPUMemoryMonitor(gpu_id)
        self._current_concurrency = self.INITIAL_CONCURRENCY
        self._target_concurrency = self.INITIAL_CONCURRENCY

        self._active_slots: Dict[int, AdaptiveJobSlot] = {}
        self._slot_lock = threading.Lock()

        self._last_ramp_up = 0.0
        self._last_backoff = 0.0
        self._consecutive_successes = 0
        self._consecutive_failures = 0

        self._shutdown = False
        self._executor: Optional[ThreadPoolExecutor] = None

        # Stats
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._oom_events = 0

    @property
    def current_concurrency(self) -> int:
        """Current target concurrency level"""
        return self._current_concurrency

    @property
    def active_jobs(self) -> int:
        """Number of currently running jobs"""
        with self._slot_lock:
            return len(self._active_slots)

    def _record_memory_before_job(self, job_id: int) -> float:
        """Record memory usage before starting a job"""
        stats = self.memory_monitor.get_memory_stats()
        return stats.used_mb if stats else 0.0

    def _record_job_completion(self, job_id: int, success: bool, memory_before: float):
        """Record job completion and update memory estimates"""
        stats = self.memory_monitor.get_memory_stats()
        memory_after = stats.used_mb if stats else memory_before

        # Estimate memory used by this job (peak during processing)
        # This is approximate but helps tune future estimates
        with self._slot_lock:
            slot = self._active_slots.get(job_id)
            if slot:
                # Use the difference as a rough estimate
                # Note: This is imprecise due to shared memory, but useful for trending
                estimated_usage = max(0, memory_after - memory_before + 1024)  # Add buffer
                self.memory_monitor.record_job_memory_usage(estimated_usage)

        if success:
            self._jobs_completed += 1
            self._consecutive_successes += 1
            self._consecutive_failures = 0

            # Consider ramping up if we've had consistent success
            if self._consecutive_successes >= 3:
                self._consider_ramp_up()
        else:
            self._jobs_failed += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0

            # Back off on failures
            if self._consecutive_failures >= 2:
                self._force_backoff("consecutive failures")

    def _consider_ramp_up(self):
        """Consider increasing concurrency if conditions are right"""
        now = time.time()

        # Don't ramp up too quickly
        if now - self._last_ramp_up < self.RAMP_UP_INTERVAL:
            return

        # Don't ramp up if we recently backed off
        if now - self._last_backoff < self.BACKOFF_COOLDOWN:
            return

        # Check if memory allows more jobs
        recommended = self.memory_monitor.get_safe_concurrent_jobs(self.active_jobs)

        if recommended > self._current_concurrency and self._current_concurrency < self.max_concurrency:
            new_concurrency = min(self._current_concurrency + 1, self.max_concurrency, recommended)
            logger.info(
                f"GPU {self.gpu_id}: Ramping up concurrency {self._current_concurrency} -> {new_concurrency} "
                f"(memory allows {recommended}, {self._consecutive_successes} consecutive successes)"
            )
            self._current_concurrency = new_concurrency
            self._last_ramp_up = now
            self._consecutive_successes = 0

    def _force_backoff(self, reason: str):
        """Force reduce concurrency"""
        if self._current_concurrency > 1:
            old = self._current_concurrency
            self._current_concurrency = max(1, self._current_concurrency - 1)
            self._last_backoff = time.time()
            logger.warning(
                f"GPU {self.gpu_id}: Backing off concurrency {old} -> {self._current_concurrency} "
                f"(reason: {reason})"
            )

            # Force memory cleanup
            self.memory_monitor.force_memory_cleanup()

    def _handle_oom(self):
        """Handle an OOM event"""
        self._oom_events += 1
        logger.error(f"GPU {self.gpu_id}: OOM detected! Total OOM events: {self._oom_events}")

        # Aggressive backoff on OOM
        self._current_concurrency = 1
        self._last_backoff = time.time()

        # Force cleanup
        self.memory_monitor.force_memory_cleanup()

        # Extra wait for memory to settle
        time.sleep(5.0)

    def _memory_watchdog(self):
        """Background thread that monitors memory and adjusts concurrency"""
        logger.info(f"GPU {self.gpu_id}: Memory watchdog started")

        while not self._shutdown:
            try:
                stats = self.memory_monitor.get_memory_stats()

                if stats:
                    # Log memory status periodically
                    logger.debug(
                        f"GPU {self.gpu_id}: Memory {stats.used_mb:.0f}/{stats.total_mb:.0f}MB "
                        f"({stats.utilization_pct:.1f}% used), {self.active_jobs} active jobs, "
                        f"concurrency={self._current_concurrency}"
                    )

                    # Check for memory pressure
                    if stats.is_very_low:
                        logger.warning(
                            f"GPU {self.gpu_id}: Critical memory pressure! "
                            f"{stats.free_mb:.0f}MB free ({100-stats.utilization_pct:.1f}%)"
                        )
                        self._force_backoff("critical memory pressure")
                    elif stats.is_low and self.active_jobs > 1:
                        logger.info(
                            f"GPU {self.gpu_id}: Memory pressure detected, "
                            f"will not start new jobs until current complete"
                        )

                time.sleep(self.MEMORY_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"GPU {self.gpu_id}: Watchdog error: {e}")
                time.sleep(self.MEMORY_CHECK_INTERVAL)

        logger.info(f"GPU {self.gpu_id}: Memory watchdog stopped")

    def _process_job_wrapper(self, job: Dict) -> bool:
        """Wrapper that handles memory tracking and OOM detection"""
        job_id = job.get("job_id", 0)
        memory_before = self._record_memory_before_job(job_id)

        # Create slot entry
        slot = AdaptiveJobSlot(
            job_id=job_id,
            memory_before_mb=memory_before
        )

        with self._slot_lock:
            self._active_slots[job_id] = slot

        try:
            # Actually process the job
            success = self.job_processor(job)
            self._record_job_completion(job_id, success, memory_before)
            return success

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU {self.gpu_id}: OOM during job #{job_id}: {e}")
            self._handle_oom()
            self._record_job_completion(job_id, False, memory_before)
            return False

        except RuntimeError as e:
            # CUDA errors often manifest as RuntimeError
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.error(f"GPU {self.gpu_id}: CUDA error during job #{job_id}: {e}")
                self._handle_oom()
                self._record_job_completion(job_id, False, memory_before)
                return False
            raise

        finally:
            with self._slot_lock:
                self._active_slots.pop(job_id, None)

    def can_accept_job(self) -> bool:
        """Check if scheduler can accept a new job"""
        if self._shutdown:
            return False

        # Check if we're at concurrency limit
        if self.active_jobs >= self._current_concurrency:
            return False

        # Check memory
        if not self.memory_monitor.can_start_new_job():
            logger.debug(f"GPU {self.gpu_id}: Cannot accept job - insufficient memory")
            return False

        # Check for throttling due to memory pressure
        if self.memory_monitor.should_throttle():
            logger.debug(f"GPU {self.gpu_id}: Cannot accept job - throttled due to memory pressure")
            return False

        return True

    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        stats = self.memory_monitor.get_memory_stats()
        return {
            "gpu_id": self.gpu_id,
            "current_concurrency": self._current_concurrency,
            "max_concurrency": self.max_concurrency,
            "active_jobs": self.active_jobs,
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "oom_events": self._oom_events,
            "memory_used_mb": stats.used_mb if stats else None,
            "memory_total_mb": stats.total_mb if stats else None,
            "memory_utilization_pct": stats.utilization_pct if stats else None,
        }


class SharedWorkerStats:
    """
    Shared statistics collector for multi-process workers.

    Uses multiprocessing.Manager to share stats across processes.
    Each GPU worker updates its own stats, and the main process
    aggregates them for display.
    """

    def __init__(self, manager: Optional[Manager] = None):
        """
        Initialize shared stats.

        Args:
            manager: Optional multiprocessing.Manager instance.
                     If None, creates a new one (use for single-process mode).
        """
        self._own_manager = manager is None
        self._manager = manager or Manager()

        # Shared dict for per-GPU stats
        # Key: gpu_id, Value: dict of stats
        self._gpu_stats = self._manager.dict()

        # Global counters (shared across all processes)
        self._global = self._manager.dict({
            "total_completed": 0,
            "total_failed": 0,
            "total_oom_events": 0,
            "workers_spawned": 0,
            "start_time": time.time(),
        })

        # Lock for atomic updates
        self._lock = self._manager.Lock()

    def update_gpu_stats(self, gpu_id: int, stats: Dict[str, Any]):
        """Update stats for a specific GPU"""
        with self._lock:
            self._gpu_stats[gpu_id] = {
                **stats,
                "last_update": time.time(),
            }

    def increment_completed(self, count: int = 1):
        """Increment global completed counter"""
        with self._lock:
            self._global["total_completed"] = self._global.get("total_completed", 0) + count

    def increment_failed(self, count: int = 1):
        """Increment global failed counter"""
        with self._lock:
            self._global["total_failed"] = self._global.get("total_failed", 0) + count

    def increment_oom(self, count: int = 1):
        """Increment global OOM counter"""
        with self._lock:
            self._global["total_oom_events"] = self._global.get("total_oom_events", 0) + count

    def get_all_stats(self) -> Dict[str, Any]:
        """Get aggregated stats from all GPUs"""
        with self._lock:
            gpu_stats = dict(self._gpu_stats)
            global_stats = dict(self._global)

        # Calculate aggregates
        total_active = sum(s.get("active_jobs", 0) for s in gpu_stats.values())
        total_concurrency = sum(s.get("current_concurrency", 0) for s in gpu_stats.values())
        max_concurrency = sum(s.get("max_concurrency", 0) for s in gpu_stats.values())

        # Memory stats
        total_memory_used = sum(s.get("memory_used_mb", 0) or 0 for s in gpu_stats.values())
        total_memory_total = sum(s.get("memory_total_mb", 0) or 0 for s in gpu_stats.values())
        avg_memory_pct = (
            (total_memory_used / total_memory_total * 100)
            if total_memory_total > 0 else 0
        )

        # Uptime
        uptime_seconds = time.time() - global_stats.get("start_time", time.time())

        # Throughput
        total_completed = global_stats.get("total_completed", 0)
        jobs_per_minute = (total_completed / uptime_seconds * 60) if uptime_seconds > 0 else 0

        return {
            "gpu_count": len(gpu_stats),
            "gpu_stats": gpu_stats,
            "total_active_jobs": total_active,
            "total_concurrency": total_concurrency,
            "max_concurrency": max_concurrency,
            "total_completed": total_completed,
            "total_failed": global_stats.get("total_failed", 0),
            "total_oom_events": global_stats.get("total_oom_events", 0),
            "total_memory_used_mb": total_memory_used,
            "total_memory_total_mb": total_memory_total,
            "avg_memory_utilization_pct": avg_memory_pct,
            "uptime_seconds": uptime_seconds,
            "jobs_per_minute": jobs_per_minute,
        }

    def cleanup(self):
        """Cleanup manager if we own it"""
        if self._own_manager and self._manager:
            try:
                self._manager.shutdown()
            except:
                pass


def print_global_stats_rich(stats: Dict[str, Any]):
    """
    Print global stats using Rich for colorful output.

    This is called from the main process to display aggregated stats.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich import box
    except ImportError:
        # Fallback to plain text if Rich not available
        logger.info(
            f"Stats: {stats['total_completed']} completed, "
            f"{stats['total_failed']} failed, "
            f"{stats['total_active_jobs']}/{stats['total_concurrency']} active, "
            f"{stats['jobs_per_minute']:.1f} jobs/min"
        )
        return

    console = Console()

    # Format uptime
    uptime = stats["uptime_seconds"]
    hours, remainder = divmod(int(uptime), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Create main stats table
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_column("Label", style="dim")
    table.add_column("Value", justify="right")
    table.add_column("Label2", style="dim")
    table.add_column("Value2", justify="right")

    # Row 1: Jobs completed / failed
    completed_style = "bold green" if stats["total_completed"] > 0 else "white"
    failed_style = "bold red" if stats["total_failed"] > 0 else "dim"
    table.add_row(
        "Completed", f"[{completed_style}]{stats['total_completed']}[/]",
        "Failed", f"[{failed_style}]{stats['total_failed']}[/]"
    )

    # Row 2: Workers / throughput
    workers_str = f"{stats['total_active_jobs']}/{stats['total_concurrency']}"
    throughput_str = f"{stats['jobs_per_minute']:.2f}/min"
    table.add_row(
        "Workers", f"[cyan]{workers_str}[/]",
        "Throughput", f"[magenta]{throughput_str}[/]"
    )

    # Row 3: Memory / OOMs
    mem_pct = stats["avg_memory_utilization_pct"]
    if mem_pct > 90:
        mem_style = "bold red"
    elif mem_pct > 75:
        mem_style = "yellow"
    else:
        mem_style = "green"
    mem_str = f"{stats['total_memory_used_mb']:.0f}/{stats['total_memory_total_mb']:.0f}MB ({mem_pct:.1f}%)"

    oom_style = "bold red" if stats["total_oom_events"] > 0 else "dim green"
    table.add_row(
        "GPU Memory", f"[{mem_style}]{mem_str}[/]",
        "OOM Events", f"[{oom_style}]{stats['total_oom_events']}[/]"
    )

    # Row 4: GPUs / Uptime
    table.add_row(
        "GPUs", f"[blue]{stats['gpu_count']}[/]",
        "Uptime", f"[dim]{uptime_str}[/]"
    )

    # Per-GPU details (if multiple GPUs)
    gpu_details = []
    for gpu_id, gpu_stat in sorted(stats.get("gpu_stats", {}).items()):
        mem_pct = gpu_stat.get("memory_utilization_pct", 0) or 0
        if mem_pct > 90:
            mem_color = "red"
        elif mem_pct > 75:
            mem_color = "yellow"
        else:
            mem_color = "green"

        active = gpu_stat.get("active_jobs", 0)
        concurrency = gpu_stat.get("current_concurrency", 1)
        completed = gpu_stat.get("jobs_completed", 0)

        gpu_details.append(
            f"[bold]GPU {gpu_id}[/]: [{mem_color}]{mem_pct:.0f}%[/] mem, "
            f"[cyan]{active}/{concurrency}[/] active, "
            f"[green]{completed}[/] done"
        )

    # Build subtitle with per-GPU details
    subtitle = ""
    if gpu_details:
        subtitle = " │ ".join(gpu_details)

    # Print panel
    panel = Panel(
        table,
        title="[bold blue]📊 Worker Stats[/]",
        subtitle=f"[dim]{subtitle}[/]" if subtitle else None,
        border_style="blue",
        padding=(0, 1),
    )

    console.print(panel)


class GlobalStatsReporter:
    """
    Background thread/process that periodically prints global stats using Rich.
    """

    def __init__(
        self,
        shared_stats: SharedWorkerStats,
        interval: float = 30.0,
    ):
        """
        Initialize the stats reporter.

        Args:
            shared_stats: SharedWorkerStats instance to read from
            interval: How often to print stats (seconds)
        """
        self.shared_stats = shared_stats
        self.interval = interval
        self._shutdown = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the reporter thread"""
        self._shutdown = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Global stats reporter started (interval: {self.interval}s)")

    def stop(self):
        """Stop the reporter thread"""
        self._shutdown = True
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        """Reporter loop"""
        # Wait a bit before first report to let workers start
        time.sleep(min(10, self.interval))

        while not self._shutdown:
            try:
                stats = self.shared_stats.get_all_stats()
                print_global_stats_rich(stats)
            except Exception as e:
                logger.error(f"Error printing stats: {e}")

            # Sleep in small increments to check for shutdown
            for _ in range(int(self.interval)):
                if self._shutdown:
                    break
                time.sleep(1)

    def print_final_stats(self):
        """Print final stats on shutdown"""
        try:
            from rich.console import Console
            console = Console()
            console.print("\n[bold yellow]═══ Final Stats ═══[/]")
            stats = self.shared_stats.get_all_stats()
            print_global_stats_rich(stats)
        except Exception as e:
            logger.error(f"Error printing final stats: {e}")


class AudioProcessor:
    """Handles audio processing and diarization"""

    def __init__(
        self,
        config: str = "high_latency",
        model_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        languages: str = "en",
        batch_size: int = 4,
    ):
        """
        Initialize audio processor with models

        Args:
            config: Streaming configuration (very_high_latency, high_latency, low_latency, ultra_low_latency)
            model_path: Optional path to custom .nemo model file
            gpu_id: GPU device ID to use (None for auto-select)
            languages: Comma-separated language codes to determine which models to load
            batch_size: Batch size for FireRedASR transcription (1, 2, 4, 8, 16, etc.) Default: 4
        """
        self.config_name = config
        self.gpu_id = gpu_id
        self.languages = [lang.strip().lower() for lang in languages.split(",")]
        self.batch_size = batch_size
        self.configs = {
            "very_high_latency": {
                "CHUNK_SIZE": 340,
                "RIGHT_CONTEXT": 40,
                "FIFO_SIZE": 40,
                "UPDATE_PERIOD": 300,
                "SPEAKER_CACHE_SIZE": 188,
            },
            "high_latency": {
                "CHUNK_SIZE": 124,
                "RIGHT_CONTEXT": 1,
                "FIFO_SIZE": 124,
                "UPDATE_PERIOD": 124,
                "SPEAKER_CACHE_SIZE": 188,
            },
            "low_latency": {
                "CHUNK_SIZE": 6,
                "RIGHT_CONTEXT": 7,
                "FIFO_SIZE": 188,
                "UPDATE_PERIOD": 144,
                "SPEAKER_CACHE_SIZE": 188,
            },
            "ultra_low_latency": {
                "CHUNK_SIZE": 3,
                "RIGHT_CONTEXT": 1,
                "FIFO_SIZE": 188,
                "UPDATE_PERIOD": 144,
                "SPEAKER_CACHE_SIZE": 188,
            },
        }

        self.diar_model = None
        self.asr_model = None  # Parakeet (English)
        self.zh_asr_model = None  # Paraformer (Chinese)
        self.bgm_classifier = None  # BGM detection
        self.bgm_model_id = "podcasts-org/detect-background-music"  # HF model ID
        self.nisqa_predictor = None  # NISQA quality assessment
        self.model_path = model_path

    def load_models(self):
        """Load diarization and ASR models"""
        # Set GPU device if specified
        if self.gpu_id is not None:
            try:
                torch.cuda.set_device(self.gpu_id)
                logger.info(
                    f"Using GPU {self.gpu_id}: {torch.cuda.get_device_name(self.gpu_id)}"
                )
            except Exception as e:
                logger.warning(f"Could not set GPU {self.gpu_id}: {e}")

        logger.info("Loading diarization model...")

        # Determine map location for model loading
        if self.gpu_id is not None:
            map_location = f"cuda:{self.gpu_id}"
        else:
            map_location = "cpu"

        if self.model_path and os.path.exists(self.model_path):
            self.diar_model = SortformerEncLabelModel.restore_from(
                restore_path=self.model_path, map_location=map_location, strict=False
            )
        else:
            self.diar_model = SortformerEncLabelModel.from_pretrained(
                "nvidia/diar_streaming_sortformer_4spk-v2"
            )
            # Move to correct GPU if specified
            if self.gpu_id is not None:
                self.diar_model = self.diar_model.to(map_location)

        self.diar_model.eval()
        logger.info(f"✓ Diarization model loaded on {map_location}")

        # Determine which ASR models to load based on languages
        needs_chinese = "zh" in self.languages or "cn" in self.languages
        needs_english = any(lang not in ["zh", "cn"] for lang in self.languages)

        # Load Parakeet (English) if needed with FP16 optimization
        if needs_english:
            logger.info("Loading Parakeet ASR model (English) with FP16 optimization...")
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v3"
            )
            # Move to correct GPU if specified
            if self.gpu_id is not None:
                self.asr_model = self.asr_model.to(f"cuda:{self.gpu_id}")

            self.asr_model.eval()
            logger.info(f"✓ Parakeet ASR model loaded on {map_location}")

        # Load Paraformer (Chinese) if needed
        if needs_chinese:
            logger.info("Loading Paraformer model (Chinese)...")
            try:
                from funasr import AutoModel

                self.zh_asr_model = AutoModel(
                    model="paraformer-zh",
                    model_revision="v2.0.4",
                    vad_model="fsmn-vad",
                    vad_model_revision="v2.0.4",
                    punc_model="ct-punc-c",
                    punc_model_revision="v2.0.4",
                    hub="hf",
                    device=f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cpu",
                )
                logger.info(f"✓ Paraformer model loaded on {map_location}")
            except ImportError as e:
                logger.error(f"Failed to import FunASR: {e}")
                logger.error("Install with: pip install funasr")
                raise
            except Exception as e:
                logger.error(f"Failed to load Paraformer model: {e}")
                raise

        # Load BGM classifier with FP16 optimization
        logger.info("Loading BGM classifier with FP16 optimization...")
        try:
            from transformers import pipeline

            # Enable FP16 for transformers pipeline on GPU
            if torch.cuda.is_available():
                self.bgm_classifier = pipeline(
                    "audio-classification",
                    model=self.bgm_model_id,
                    device=self.gpu_id if self.gpu_id is not None else 0,
                    torch_dtype=torch.float16,  # Enable FP16 inference
                )
                logger.info(f"✓ BGM classifier loaded ({self.bgm_model_id}) (FP16 enabled)")
            else:
                self.bgm_classifier = pipeline(
                    "audio-classification",
                    model=self.bgm_model_id,
                    device=-1,  # CPU
                )
                logger.info(f"✓ BGM classifier loaded ({self.bgm_model_id})")
        except Exception as e:
            logger.error(f"Failed to load BGM classifier: {e}")
            raise

        # Load NISQA quality assessment model with FP16 optimization
        logger.info("Loading NISQA quality assessment model (FP16 optimized)...")
        try:
            device = torch.device(f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cuda" if torch.cuda.is_available() else "cpu")
            # Enable FP16 for ~2x speedup on GPU, disable torch.compile for compatibility
            self.nisqa_predictor = NISQAPredictor(
                device=device,
                dim=True,
                fp16=True,  # Enable FP16 inference
                compile_model=False  # Disabled: incompatible with pack_padded_sequence
            )
            logger.info(f"✓ NISQA model loaded on {device} (FP16: {self.nisqa_predictor.fp16})")
        except Exception as e:
            logger.error(f"Failed to load NISQA model: {e}")
            raise

        # Set streaming configuration
        config = self.configs[self.config_name]
        self.diar_model.sortformer_modules.chunk_len = config["CHUNK_SIZE"]
        self.diar_model.sortformer_modules.chunk_right_context = config["RIGHT_CONTEXT"]
        self.diar_model.sortformer_modules.fifo_len = config["FIFO_SIZE"]
        self.diar_model.sortformer_modules.spkcache_update_period = config[
            "UPDATE_PERIOD"
        ]
        self.diar_model.sortformer_modules.spkcache_len = config["SPEAKER_CACHE_SIZE"]
        self.diar_model.sortformer_modules._check_streaming_parameters()
        logger.info(f"✓ Using {self.config_name} configuration")

    def load_audio_torchaudio(self, audio_path: str, target_sr: int = 16000) -> tuple:
        """
        Load audio using torchaudio (CPU-based for thread safety in concurrent mode)

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (default: 16000)

        Returns:
            Tuple of (audio_array, sample_rate) - audio as numpy array
        """
        # Load audio with torchaudio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        # NOTE: We do resampling on CPU to avoid CUDA race conditions in concurrent mode.
        # The GPU memory saved by CPU resampling is better used for the actual ML models.
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        # Convert to numpy and squeeze to 1D
        audio_array = waveform.squeeze().cpu().numpy()

        return audio_array, sr

    @staticmethod
    def convert_to_mono_if_needed(audio_path: str) -> str:
        """Convert stereo to mono if needed"""
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        if audio.ndim > 1:
            logger.info(f"Converting {audio_path} to mono...")
            mono_path = str(Path(audio_path).with_suffix("")) + "_mono.wav"
            audio_mono = librosa.to_mono(audio)
            sf.write(mono_path, audio_mono, sr)
            return mono_path

        return audio_path

    @staticmethod
    def extract_audio_segment(audio, sr: int, start_time: float, end_time: float):
        """Extract a segment from audio array based on timestamps"""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        return audio[start_sample:end_sample]

    @staticmethod
    def compute_file_hashes(filepath: str) -> Dict[str, str]:
        """Compute SHA256 and MD5 hashes of a file"""
        sha256_hash = hashlib.sha256()
        md5_hash = hashlib.md5()

        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                md5_hash.update(byte_block)

        return {"sha256": sha256_hash.hexdigest(), "md5": md5_hash.hexdigest()}

    def diarize_audio(
        self, audio_path: str, episode_url: str = None, language: str = None,
        converted_audio_path: str = None
    ) -> Dict:
        """
        Diarize a single audio file and return results

        Args:
            audio_path: Path to original audio file (used for hashing)
            episode_url: Original episode URL
            language: Language code
            converted_audio_path: Optional pre-converted mono audio path (from prefetch)

        Returns:
            Dict with diarization results, transcriptions, file hashes, and metadata
        """
        start_time = time.time()
        # Compute hashes on the ORIGINAL audio file (before any conversion)
        logger.info(f"Computing file hashes for {audio_path}...")
        hashes = self.compute_file_hashes(audio_path)
        logger.info(f"SHA256: {hashes['sha256']}")
        logger.info(f"MD5: {hashes['md5']}")

        # Use pre-converted path if provided, otherwise convert now
        if converted_audio_path:
            logger.info(f"Using pre-converted audio: {converted_audio_path}")
            processing_audio_path = converted_audio_path
        else:
            processing_audio_path = self.convert_to_mono_if_needed(audio_path)

        # Get audio duration using torchaudio (GPU-accelerated)
        audio, sr = self.load_audio_torchaudio(processing_audio_path, target_sr=16000)
        duration = len(audio) / sr

        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

        # Perform diarization
        logger.info("Diarizing...")
        segments = self.diar_model.diarize(audio=processing_audio_path, batch_size=1)

        # Parse segments into structured format
        results = []
        segment_list = (
            segments[0]
            if isinstance(segments, list) and len(segments) > 0
            else segments
        )

        for seg in segment_list:
            if isinstance(seg, str):
                parts = seg.split()
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2].replace("speaker_", "")  # Remove 'speaker_' prefix
                results.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "duration": end - start,
                    }
                )

        # Extract audio segments IN MEMORY (no temp files for massive speedup)
        logger.info(f"Extracting {len(results)} segments...")
        segment_audio_arrays = []  # Keep in memory instead of writing to disk
        for i, segment in enumerate(results):
            segment_audio = self.extract_audio_segment(
                audio, sr, segment["start"], segment["end"]
            )

            # Compute clipping and loudness metrics for this segment
            # Clipping detection: count samples at or near maximum amplitude
            clipped_samples = np.sum(np.abs(segment_audio) >= 0.99)
            clip_rate = float(clipped_samples / len(segment_audio)) if len(segment_audio) > 0 else 0.0

            # Loudness metrics
            rms = np.sqrt(np.mean(segment_audio**2))
            rms_db = 20 * np.log10(rms) if rms > 0 else -100.0
            peak = np.max(np.abs(segment_audio))
            peak_db = 20 * np.log10(peak) if peak > 0 else -100.0

            # Add to segment metadata
            segment["clip_rate"] = clip_rate
            segment["clipped_samples"] = int(clipped_samples)
            segment["has_clipping"] = clip_rate > 0.001  # >0.1% clipping threshold
            segment["rms_db"] = float(rms_db)
            segment["peak_db"] = float(peak_db)
            segment["dynamic_range_db"] = float(peak_db - rms_db)

            # Store audio in memory (no disk I/O!)
            segment_audio_arrays.append(segment_audio)

        try:
            # Transcribe all segments at once using appropriate model (IN MEMORY)
            is_chinese = language and (
                language.lower() == "zh" or language.lower() == "cn"
            )

            logger.info(f"Transcribing {len(segment_audio_arrays)} segments...")

            # Write segments to temp files only for ASR (most ASR models require files)
            # TODO: Future optimization - use in-memory transcription if models support it
            temp_files = []
            session_id = uuid.uuid4().hex[:8]
            for i, segment_audio in enumerate(segment_audio_arrays):
                temp_path = f"/tmp/segment_{session_id}_{i}.wav"
                sf.write(temp_path, segment_audio, 16000)
                temp_files.append(temp_path)

            if is_chinese and self.zh_asr_model:
                # Use Paraformer for Chinese - process all files at once
                logger.info("Using Paraformer for Chinese transcription")

                # Pass all temp files at once
                paraformer_results = self.zh_asr_model.generate(
                    input=temp_files, batch_size_s=300
                )

                # Extract text from results - format is [{'key': 'filename', 'text': 'transcription'}, ...]
                transcriptions = []
                for result in paraformer_results:
                    if isinstance(result, dict):
                        transcriptions.append(result.get("text", ""))
                    else:
                        transcriptions.append("")

            elif self.asr_model:
                # Use Parakeet for English/other languages
                logger.info("Using Parakeet for transcription")
                batch_results = self.asr_model.transcribe(temp_files)
                transcriptions = [result.text for result in batch_results]
            else:
                logger.error("No ASR model available for this language")
                transcriptions = ["" for _ in temp_files]

            # Add transcriptions to results
            logger.info(
                f"Adding {len(transcriptions)} transcriptions to {len(results)} segments"
            )
            for i in range(len(results)):
                results[i]["transcription"] = transcriptions[i]
                if i < 3:  # Log first 3 for debugging
                    logger.debug(
                        f"Segment {i}: '{transcriptions[i][:50] if transcriptions[i] else '(empty)'}'..."
                    )

            # BGM classification for all segments (BATCHED, IN-MEMORY for GPU efficiency)
            logger.info(f"Classifying BGM for {len(segment_audio_arrays)} segments...")

            # Use in-memory audio arrays directly (NO FILE I/O!)
            audio_inputs = []
            for segment_audio in segment_audio_arrays:
                try:
                    # Ensure float32
                    if segment_audio.dtype != np.float32:
                        segment_audio = segment_audio.astype(np.float32)

                    audio_inputs.append(
                        {"array": segment_audio, "sampling_rate": 16000}
                    )
                except Exception as e:
                    logger.warning(f"Failed to prepare audio for BGM classification: {e}")
                    audio_inputs.append(None)

            # Process in batches for GPU efficiency
            try:
                # Filter out None values and track indices
                valid_inputs = []
                valid_indices = []
                for i, audio_input in enumerate(audio_inputs):
                    if audio_input is not None:
                        valid_inputs.append(audio_input)
                        valid_indices.append(i)
                    else:
                        results[i]["bgm_probability"] = 0.0
                        results[i]["bgm"] = False

                # Batch classify all valid inputs at once with larger batch size for FP16
                if valid_inputs:
                    batch_predictions = self.bgm_classifier(valid_inputs, batch_size=128)

                    # Map predictions back to results
                    for idx, predictions in zip(valid_indices, batch_predictions):
                        bgm_prob = 0.0
                        for pred in predictions:
                            if pred["label"] == "bgm":
                                bgm_prob = pred["score"]
                                break
                        results[idx]["bgm_probability"] = bgm_prob
                        results[idx]["bgm"] = bgm_prob > 0.5

                logger.info(f"✓ Classified {len(valid_inputs)} segments")
            except Exception as e:
                logger.warning(f"Batch BGM classification failed: {e}, falling back to defaults")
                for i in range(len(results)):
                    if "bgm_probability" not in results[i]:
                        results[i]["bgm_probability"] = 0.0
                        results[i]["bgm"] = False

            # NISQA quality assessment for all segments (BATCHED, IN-MEMORY for GPU efficiency)
            logger.info(f"Assessing audio quality (NISQA) for {len(segment_audio_arrays)} segments...")

            try:
                # Use segment_audio_arrays directly (already in memory!)
                valid_arrays = []
                valid_indices = []

                for i, segment_audio in enumerate(segment_audio_arrays):
                    if segment_audio is not None and len(segment_audio) > 0:
                        valid_arrays.append(segment_audio)
                        valid_indices.append(i)
                    else:
                        # Set defaults for failed segments
                        results[i]["quality_mos"] = None
                        results[i]["quality_noisiness"] = None
                        results[i]["quality_discontinuity"] = None
                        results[i]["quality_coloration"] = None
                        results[i]["quality_loudness"] = None

                if valid_arrays:
                    # Batch predict using official NISQA implementation
                    # Use larger batch size for better GPU utilization with FP16
                    predictions = self.nisqa_predictor.predict_arrays(
                        audio_arrays=valid_arrays,
                        sample_rate=16000,
                        batch_size=64  # Increased from 16 to 64 for FP16 efficiency
                    )

                    # Map predictions back to results
                    successful_assessments = 0
                    for idx, pred_idx in enumerate(valid_indices):
                        try:
                            results[pred_idx]["quality_mos"] = float(predictions['mos'][idx])
                            results[pred_idx]["quality_noisiness"] = float(predictions['noisiness'][idx])
                            results[pred_idx]["quality_discontinuity"] = float(predictions['discontinuity'][idx])
                            results[pred_idx]["quality_coloration"] = float(predictions['coloration'][idx])
                            results[pred_idx]["quality_loudness"] = float(predictions['loudness'][idx])
                            successful_assessments += 1
                        except Exception as e:
                            logger.warning(f"Failed to process NISQA result for segment {pred_idx}: {e}")
                            results[pred_idx]["quality_mos"] = None
                            results[pred_idx]["quality_noisiness"] = None
                            results[pred_idx]["quality_discontinuity"] = None
                            results[pred_idx]["quality_coloration"] = None
                            results[pred_idx]["quality_loudness"] = None

                    logger.info(f"✓ Assessed quality for {successful_assessments}/{len(valid_arrays)} segments")
                else:
                    logger.info("No valid segments for NISQA assessment")

            except Exception as e:
                logger.warning(f"Batch NISQA assessment failed: {e}, setting defaults")
                import traceback
                traceback.print_exc()
                for i in range(len(results)):
                    if "quality_mos" not in results[i]:
                        results[i]["quality_mos"] = None
                        results[i]["quality_noisiness"] = None
                        results[i]["quality_discontinuity"] = None
                        results[i]["quality_coloration"] = None
                        results[i]["quality_loudness"] = None

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Get GPU info
        gpu_info = get_gpu_info(self.gpu_id)

        # Compute episode-level quality statistics from segment scores
        quality_mos_scores = [s.get("quality_mos") for s in results if s.get("quality_mos") is not None]
        quality_noisiness_scores = [s.get("quality_noisiness") for s in results if s.get("quality_noisiness") is not None]
        quality_discontinuity_scores = [s.get("quality_discontinuity") for s in results if s.get("quality_discontinuity") is not None]
        quality_coloration_scores = [s.get("quality_coloration") for s in results if s.get("quality_coloration") is not None]
        quality_loudness_scores = [s.get("quality_loudness") for s in results if s.get("quality_loudness") is not None]

        # Clipping and loudness aggregations (per-segment values)
        clip_rates = [s.get("clip_rate", 0.0) for s in results]
        rms_db_values = [s.get("rms_db") for s in results if s.get("rms_db") is not None and s.get("rms_db") != -100.0]
        peak_db_values = [s.get("peak_db") for s in results if s.get("peak_db") is not None and s.get("peak_db") != -100.0]
        dynamic_range_values = [s.get("dynamic_range_db") for s in results if s.get("dynamic_range_db") is not None]

        episode_quality = {}
        if quality_mos_scores:
            episode_quality = {
                # NISQA scores
                "mean_mos": float(np.mean(quality_mos_scores)),
                "median_mos": float(np.median(quality_mos_scores)),
                "p25_mos": float(np.percentile(quality_mos_scores, 25)),
                "p75_mos": float(np.percentile(quality_mos_scores, 75)),
                "min_mos": float(np.min(quality_mos_scores)),
                "max_mos": float(np.max(quality_mos_scores)),
                "mean_noisiness": float(np.mean(quality_noisiness_scores)),
                "mean_discontinuity": float(np.mean(quality_discontinuity_scores)),
                "mean_coloration": float(np.mean(quality_coloration_scores)),
                "mean_loudness": float(np.mean(quality_loudness_scores)),
                # Quality tier counts
                "high_quality_segments": sum(1 for s in quality_mos_scores if s >= 4.0),
                "medium_quality_segments": sum(1 for s in quality_mos_scores if 3.0 <= s < 4.0),
                "low_quality_segments": sum(1 for s in quality_mos_scores if s < 3.0),
            }

        # Clipping statistics
        clipping_stats = {
            "mean_clip_rate": float(np.mean(clip_rates)) if clip_rates else 0.0,
            "max_clip_rate": float(np.max(clip_rates)) if clip_rates else 0.0,
            "segments_with_clipping": sum(1 for s in results if s.get("has_clipping", False)),
            "total_clipped_samples": sum(s.get("clipped_samples", 0) for s in results),
        }

        # Loudness statistics
        loudness_stats = {}
        if rms_db_values:
            loudness_stats = {
                "mean_rms_db": float(np.mean(rms_db_values)),
                "median_rms_db": float(np.median(rms_db_values)),
                "min_rms_db": float(np.min(rms_db_values)),
                "max_rms_db": float(np.max(rms_db_values)),
                "mean_peak_db": float(np.mean(peak_db_values)),
                "max_peak_db": float(np.max(peak_db_values)),
                "mean_dynamic_range_db": float(np.mean(dynamic_range_values)),
                "rms_variation": float(np.std(rms_db_values)),  # Loudness consistency
            }

        # Create output record with all metadata
        output_record = {
            "audio_filepath": str(Path(audio_path).absolute()),
            "episode_url": episode_url,
            "language": language,
            "duration": duration,
            "num_segments": len(results),
            "segments": results,
            "file_hashes": hashes,
            "num_speakers": len(set(s["speaker"] for s in results)),
            "processing_time": processing_time,
            "gpu_info": gpu_info,
            "bgm_model": self.bgm_model_id,
            "worker_version": WORKER_VERSION,
            "episode_quality": episode_quality,
            "clipping_stats": clipping_stats,
            "loudness_stats": loudness_stats,
            "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        logger.info(f"✓ Processed {len(results)} segments in {processing_time:.2f}s")
        logger.info(f"  Speakers detected: {output_record['num_speakers']}")
        if episode_quality:
            logger.info(f"  Quality: MOS={episode_quality['mean_mos']:.2f} (median={episode_quality['median_mos']:.2f})")
        if clipping_stats["segments_with_clipping"] > 0:
            logger.info(f"  Clipping: {clipping_stats['segments_with_clipping']}/{len(results)} segments, max rate={clipping_stats['max_clip_rate']:.4f}")
        if loudness_stats:
            logger.info(f"  Loudness: mean={loudness_stats['mean_rms_db']:.1f}dB, peak={loudness_stats['max_peak_db']:.1f}dB")
        if gpu_info:
            logger.info(f"  GPU: {gpu_info}")

        return output_record


class S3Uploader:
    """Handles S3 uploads in a background thread"""

    def __init__(self, s3_config: Dict):
        """
        Initialize S3 uploader

        Args:
            s3_config: Dict with keys: endpoint_url, access_key_id, secret_access_key, bucket, region
        """
        import boto3
        from botocore.client import Config

        self.bucket = s3_config["bucket"]
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=s3_config["endpoint_url"],
            aws_access_key_id=s3_config["access_key_id"],
            aws_secret_access_key=s3_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
            region_name=s3_config["region"],
        )
        logger.info(f"S3 uploader initialized (bucket: {self.bucket})")

    def compress_audio(self, input_path: str) -> str:
        """
        Compress audio to MP3 with fast, reasonable settings for podcasts

        Args:
            input_path: Path to input audio file

        Returns:
            Path to compressed MP3 file
        """
        import subprocess

        # Create compressed filename with UUID to avoid conflicts
        base_path = Path(input_path).with_suffix("")
        unique_id = uuid.uuid4().hex[:8]  # Use first 8 chars of UUID for brevity
        compressed_path = f"{base_path}_compressed_{unique_id}.mp3"

        # Use ffmpeg with fast preset, variable bitrate optimized for voice
        # -q:a 4 gives VBR ~128kbps average (good for podcasts, not music)
        # -preset fast prioritizes speed over compression efficiency
        # -ac 1 converts to mono (podcasts rarely need stereo)
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-q:a", "4",  # VBR quality (~128kbps, good for voice)
            "-ac", "1",   # Mono
            "-ar", "44100",  # 44.1kHz sample rate
            "-y",  # Overwrite output file
            "-loglevel", "error",  # Only show errors
            compressed_path
        ]

        logger.info(f"Compressing {Path(input_path).name} to MP3...")
        subprocess.run(cmd, check=True, capture_output=True)

        # Log size reduction
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(compressed_path)
        reduction = (1 - compressed_size / original_size) * 100
        logger.info(f"✓ Compressed: {original_size/(1024*1024):.1f}MB → {compressed_size/(1024*1024):.1f}MB ({reduction:.0f}% reduction)")

        return compressed_path

    def upload_file_threaded(self, file_path: str, file_hash: str) -> threading.Thread:
        """
        Compress and upload file to S3 in a background thread using SHA256 hash for organization

        Args:
            file_path: Local path to file
            file_hash: SHA256 hash of the file

        Returns:
            Thread object that is uploading the file
        """
        # Use first 3 characters of hash for subfolder to avoid too many files in one folder
        # With 3 hex chars, we get 4096 possible subfolders (16^3)
        subfolder = file_hash[:3]
        filename = Path(file_path).name
        # Use .mp3 extension since we'll compress to MP3
        base_name = Path(filename).stem
        object_name = f"{subfolder}/{file_hash}_{base_name}.mp3"

        def upload_task():
            compressed_path = None
            upload_path = file_path
            upload_object_name = object_name

            try:
                # Try to compress audio (this happens in background, not blocking GPU)
                try:
                    compressed_path = self.compress_audio(file_path)
                    upload_path = compressed_path
                except Exception as compress_error:
                    logger.warning(f"Compression failed: {compress_error}")
                    logger.info("Falling back to uploading original file")
                    # Use original filename extension for fallback
                    original_ext = Path(filename).suffix
                    upload_object_name = f"{subfolder}/{file_hash}_{Path(filename).stem}{original_ext}"
                    upload_path = file_path

                logger.info(f"Uploading {Path(upload_path).name} to S3 as {upload_object_name}...")
                self.s3_client.upload_file(upload_path, self.bucket, upload_object_name)
                url = f"{self.s3_client.meta.endpoint_url}/{self.bucket}/{upload_object_name}"
                logger.info(f"✓ Uploaded to S3: {url}")

                # Delete both original and compressed files after successful upload
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"✓ Deleted original file: {file_path}")
                if compressed_path and os.path.exists(compressed_path):
                    os.remove(compressed_path)
                    logger.info(f"✓ Deleted compressed file: {compressed_path}")
            except Exception as e:
                logger.error(f"✗ Error uploading {filename} to S3: {e}")
                # Clean up compressed file on error
                if compressed_path and os.path.exists(compressed_path):
                    try:
                        os.remove(compressed_path)
                    except:
                        pass

        thread = threading.Thread(target=upload_task, daemon=True)
        thread.start()
        return thread


class PodcastPileWorker:
    """Worker that fetches jobs and processes them"""

    def __init__(
        self,
        manager_url: str,
        worker_id: str,
        worker_password: Optional[str] = None,
        config: str = "high_latency",
        model_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        languages: str = "en",
        batch_size: int = 4,
        s3_config: Optional[Dict] = None,
    ):
        """
        Initialize worker

        Args:
            manager_url: URL of the manager server
            worker_id: Unique identifier for this worker
            worker_password: Optional password for worker authentication
            config: Diarization config (very_high_latency, high_latency, low_latency, ultra_low_latency)
            model_path: Optional path to custom .nemo model file
            gpu_id: GPU device ID to use (None for auto-select)
            languages: Comma-separated language codes worker will process
            batch_size: Batch size for FireRedASR transcription (default: 4)
            s3_config: Optional S3 configuration for audio uploads
        """
        self.manager_url = manager_url.rstrip("/")
        self.worker_id = worker_id
        self.worker_password = worker_password
        self.gpu_id = gpu_id
        self.processor = AudioProcessor(
            config=config,
            model_path=model_path,
            gpu_id=gpu_id,
            languages=languages,
            batch_size=batch_size,
        )

        # Initialize S3 uploader if config provided
        self.s3_uploader = S3Uploader(s3_config) if s3_config else None

        # Lock for GPU operations - NeMo models are NOT thread-safe
        # This ensures only one job uses GPU at a time
        self._gpu_lock = threading.Lock()

        # Headers for API requests
        self.headers = {}
        if worker_password:
            self.headers["X-Worker-Password"] = worker_password

    def load_models(self):
        """Load processing models"""
        self.processor.load_models()

    def request_job(self, languages: str = "en") -> Optional[Dict]:
        """
        Request a job from the manager

        Args:
            languages: Comma-separated language codes (default: "en" for English only)

        Returns:
            Job dict or None if no jobs available
        """
        url = f"{self.manager_url}/api/jobs/request"
        params = {"worker_id": self.worker_id, "languages": languages}

        try:
            response = requests.post(url, params=params, headers=self.headers)

            if response.status_code == 404:
                logger.info("No jobs available")
                return None

            response.raise_for_status()
            job = response.json()
            logger.info(f"Received job #{job['job_id']}: {job['episode_url']}")
            return job

        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting job: {e}")
            return None

    def download_audio(self, url: str, temp_dir: str) -> str:
        """
        Download audio file from URL

        Args:
            url: Audio file URL
            temp_dir: Directory to save file

        Returns:
            Path to downloaded file
        """
        logger.info(f"Downloading audio from {url}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Determine file extension from URL or content-type
        ext = ".mp3"  # Default
        if url.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            ext = Path(url).suffix

        filepath = os.path.join(temp_dir, f"audio{ext}")

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✓ Downloaded to {filepath}")
        return filepath

    def update_job_status(self, job_id: int, status: str) -> bool:
        """Update job status on manager"""
        # Use the /start endpoint when marking as processing
        if status == "processing":
            url = f"{self.manager_url}/api/jobs/{job_id}/start"
            params = {"worker_id": self.worker_id}
        else:
            # For other statuses, we don't have a generic endpoint
            # Just skip the update since complete/fail have their own endpoints
            return True

        try:
            response = requests.post(url, params=params, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Updated job #{job_id} status to {status}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error updating job status: {e}")
            return False

    def submit_results(self, job_id: int, results: Dict) -> bool:
        """
        Submit processing results to manager

        Args:
            job_id: Job ID
            results: Processing results dict

        Returns:
            True if successful
        """
        url = f"{self.manager_url}/api/jobs/{job_id}/complete"

        # Format the results for submission
        payload = {
            "result_json": json.dumps(results),
            "transcription": self._format_transcription(results),
            "diarization": self._format_diarization(results),
            "processing_duration": results.get("processing_time"),
            "worker_gpu": results.get("gpu_info"),
            "processed_at": results.get("processed_at"),
        }

        params = {"worker_id": self.worker_id}

        try:
            response = requests.post(
                url, params=params, json=payload, headers=self.headers
            )
            response.raise_for_status()
            logger.info(f"✓ Submitted results for job #{job_id}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error submitting results: {e}")
            try:
                logger.error(
                    f"Response body: {e.response.text if e.response else 'N/A'}"
                )
            except:
                pass
            return False

    def report_failure(self, job_id: int, error_message: str) -> bool:
        """Report job failure to manager"""
        url = f"{self.manager_url}/api/jobs/{job_id}/fail"
        params = {"worker_id": self.worker_id, "error_message": error_message}

        try:
            response = requests.post(url, params=params, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Reported failure for job #{job_id}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error reporting failure: {e}")
            return False

    @staticmethod
    def _format_transcription(results: Dict) -> str:
        """Format transcription as plain text"""
        lines = []
        for segment in results.get("segments", []):
            speaker = segment["speaker"]
            text = segment["transcription"]
            lines.append(f"Speaker {speaker}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _format_diarization(results: Dict) -> str:
        """Format diarization timestamps as plain text"""
        lines = []
        for segment in results.get("segments", []):
            start = segment["start"]
            end = segment["end"]
            speaker = segment["speaker"]
            lines.append(f"{start:.2f} {end:.2f} speaker_{speaker}")
        return "\n".join(lines)

    def process_job(self, job: Dict, next_job: Optional[Dict] = None) -> bool:
        """
        Process a single job, optionally prefetching next job's audio

        Args:
            job: Job dict from manager
            next_job: Optional next job to prefetch audio for

        Returns:
            True if successful
        """
        job_id = job["job_id"]
        episode_url = job["episode_url"]
        language = job.get("language")

        logger.info(f"Processing job #{job_id}...")

        # Update status to processing
        self.update_job_status(job_id, "processing")

        temp_dir = tempfile.mkdtemp()
        upload_thread = None

        # Prefetch state for next job
        next_temp_dir = None
        next_audio_path = None
        next_download_thread = None

        try:
            # Check if audio was prefetched
            if job.get("_prefetch_audio_path"):
                # Wait for prefetch to complete
                prefetch_thread = job.get("_prefetch_thread")
                if prefetch_thread and prefetch_thread.is_alive():
                    logger.info("Waiting for prefetched download to complete...")
                    prefetch_thread.join()

                prefetch_data = job.get("_prefetch_audio_path")
                temp_dir = job.get("_prefetch_temp_dir")

                # Handle both tuple (new format) and string (old format for backward compatibility)
                if isinstance(prefetch_data, tuple):
                    original_audio_path, audio_path = prefetch_data
                else:
                    # Old format: single path (both are the same)
                    original_audio_path = prefetch_data
                    audio_path = prefetch_data

                if audio_path and os.path.exists(audio_path):
                    logger.info(f"✓ Using prefetched audio: {audio_path}")
                else:
                    # Prefetch failed, download normally
                    logger.warning("Prefetch failed, downloading audio now...")
                    temp_dir = tempfile.mkdtemp()
                    audio_path = self.download_audio(episode_url, temp_dir)
                    original_audio_path = audio_path
            else:
                # No prefetch, download normally
                audio_path = self.download_audio(episode_url, temp_dir)
                original_audio_path = audio_path

            # Start S3 upload in background if configured
            if self.s3_uploader:
                # Compute hash for the ORIGINAL audio file (not converted)
                file_hash = self.processor.compute_file_hashes(original_audio_path)["sha256"]
                upload_thread = self.s3_uploader.upload_file_threaded(
                    original_audio_path, file_hash
                )
                logger.info("S3 upload started in background thread")

            # Process audio with metadata (GPU work happens here)
            # Pass original path for hashing and converted path for processing (if available from prefetch)
            converted_path = audio_path if audio_path != original_audio_path else None
            results = self.processor.diarize_audio(
                original_audio_path,
                episode_url=episode_url,
                language=language,
                converted_audio_path=converted_path
            )

            # Start downloading next job's audio BEFORE submitting results
            # This overlaps download I/O with result submission network I/O
            if next_job:
                next_temp_dir = tempfile.mkdtemp()
                next_episode_url = next_job["episode_url"]

                def download_next():
                    nonlocal next_audio_path
                    try:
                        logger.info(f"⏩ Prefetching audio for job #{next_job['job_id']}...")
                        original_audio_path = self.download_audio(next_episode_url, next_temp_dir)
                        logger.info(f"✓ Prefetched audio for job #{next_job['job_id']}")

                        # Pre-convert to mono if needed (CPU work during I/O time)
                        converted_audio_path = self.processor.convert_to_mono_if_needed(original_audio_path)
                        logger.info(f"✓ Pre-converted audio to mono if needed")

                        # Store both paths: we need the converted for processing, original for hashing
                        next_audio_path = (original_audio_path, converted_audio_path)
                    except Exception as e:
                        logger.warning(f"Failed to prefetch audio: {e}")

                next_download_thread = threading.Thread(target=download_next, daemon=True)
                next_download_thread.start()

            # Submit results (network I/O happens here)
            success = self.submit_results(job_id, results)

            # Wait for S3 upload to complete before finishing
            if upload_thread:
                logger.info("Waiting for S3 upload to complete...")
                upload_thread.join()
                logger.info("S3 upload thread finished")

            # Store prefetch results for next iteration
            if next_download_thread:
                # Attach prefetch info to the job object for next iteration
                next_job["_prefetch_temp_dir"] = next_temp_dir
                next_job["_prefetch_audio_path"] = next_audio_path
                next_job["_prefetch_thread"] = next_download_thread

            return success

        except Exception as e:
            logger.error(f"Error processing job #{job_id}: {e}", exc_info=True)
            self.report_failure(job_id, str(e))
            # Clean up next job prefetch on error
            if next_temp_dir:
                import shutil
                shutil.rmtree(next_temp_dir, ignore_errors=True)
            return False

        finally:
            # Cleanup temp directory (but audio file may already be deleted by S3 uploader)
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def run_once(self, languages: str = "en") -> bool:
        """
        Request and process a single job

        Args:
            languages: Comma-separated language codes (default: "en")

        Returns:
            True if job was processed, False if no job available
        """
        job = self.request_job(languages=languages)

        if not job:
            return False

        self.process_job(job)
        return True

    def run_loop(self, languages: str = "en", poll_interval: int = 10):
        """
        Continuously request and process jobs with prefetching

        Args:
            languages: Comma-separated language codes (default: "en")
            poll_interval: Seconds to wait between requests if no job available
        """
        import time

        logger.info(f"Starting worker loop (languages: {languages})...")
        logger.info("Press Ctrl+C to stop")

        next_job = None
        prefetch_thread = None

        def prefetch_job():
            """Fetch next job in background"""
            return self.request_job(languages=languages)

        try:
            while True:
                # Use prefetched job if available, otherwise fetch now
                if next_job:
                    job = next_job
                    logger.info(f"Using prefetched job #{job['job_id']}")
                    next_job = None
                else:
                    job = self.request_job(languages=languages)

                if job:
                    # Start prefetching next job in background thread
                    # This overlaps network I/O with GPU processing
                    prefetch_thread = threading.Thread(target=lambda: None, daemon=True)

                    def fetch_wrapper():
                        nonlocal next_job
                        next_job = prefetch_job()
                        if next_job:
                            logger.info(f"✓ Prefetched job #{next_job['job_id']}")

                    prefetch_thread = threading.Thread(target=fetch_wrapper, daemon=True)
                    prefetch_thread.start()

                    # Wait for prefetch to complete before processing
                    # This ensures we can start downloading next audio ASAP
                    if prefetch_thread:
                        prefetch_thread.join(timeout=5)  # Don't wait forever

                    # Process current job, passing next job for audio prefetching
                    self.process_job(job, next_job=next_job)
                else:
                    logger.info(f"No jobs available, waiting {poll_interval}s...")
                    time.sleep(poll_interval)
                    next_job = None  # Clear any stale prefetch

        except KeyboardInterrupt:
            logger.info("\nStopping worker...")

    def run_loop_adaptive(
        self,
        languages: str = "en",
        poll_interval: int = 10,
        max_concurrency: int = 4,
        shared_stats: Optional['SharedWorkerStats'] = None,
    ):
        """
        Continuously request and process jobs with adaptive concurrency using subprocesses.

        This mode spawns separate worker PROCESSES (not threads) to run concurrent jobs.
        Each subprocess loads its own copy of the models, which is required because
        NeMo/PyTorch models are not thread-safe for concurrent GPU execution.

        The scheduler:
        - Starts conservative with 1 worker process
        - Monitors GPU memory and spawns more workers when VRAM allows
        - Each worker has its own model copy (~X GB VRAM per worker)
        - Backs off by not spawning new workers when memory is tight
        - Learns typical memory usage per worker over time

        Args:
            languages: Comma-separated language codes (default: "en")
            poll_interval: Seconds to wait between requests if no job available
            max_concurrency: Maximum concurrent worker processes (default: 4)
            shared_stats: Optional SharedWorkerStats for cross-process stats reporting
        """
        logger.info(f"Starting ADAPTIVE worker loop with SUBPROCESS workers...")
        logger.info(f"Languages: {languages}, Max workers: {max_concurrency}")
        logger.info("Press Ctrl+C to stop")

        gpu_id = self.gpu_id if self.gpu_id is not None else 0
        memory_monitor = GPUMemoryMonitor(gpu_id)

        # Get baseline memory (models already loaded in this process)
        baseline_stats = memory_monitor.get_memory_stats()
        if baseline_stats:
            baseline_vram = baseline_stats.used_mb
            total_vram = baseline_stats.total_mb
            logger.info(f"Baseline VRAM usage (models loaded): {baseline_vram:.0f}MB / {total_vram:.0f}MB")
            # Estimate per-worker VRAM (model size)
            estimated_worker_vram = baseline_vram * 0.9  # Slightly less due to shared CUDA context
        else:
            estimated_worker_vram = 8000  # Conservative 8GB default
            total_vram = 24000

        logger.info(f"Estimated VRAM per worker: ~{estimated_worker_vram:.0f}MB")

        # Safety buffer - always keep this much free
        safety_buffer_mb = 2000  # 2GB safety buffer

        # Track active worker processes
        active_workers: Dict[int, multiprocessing.Process] = {}  # worker_id -> Process
        worker_counter = 0

        # Shared queue for job results (success/failure counts)
        result_queue = multiprocessing.Queue()

        # Current target concurrency (starts at 1, adapts based on memory)
        current_target = 1
        last_ramp_check = time.time()
        ramp_check_interval = 30.0  # Check for ramp-up every 30s
        consecutive_successes = 0

        def get_safe_worker_count() -> int:
            """Calculate how many workers we can safely run based on VRAM"""
            stats = memory_monitor.get_memory_stats()
            if not stats:
                return 1

            available = stats.free_mb - safety_buffer_mb
            if available <= 0:
                return max(1, len(active_workers))  # Don't kill existing workers

            # How many MORE workers could we fit?
            additional = int(available / estimated_worker_vram)
            total_possible = len(active_workers) + additional

            return min(total_possible, max_concurrency)

        def spawn_worker(worker_num: int) -> multiprocessing.Process:
            """Spawn a new worker subprocess"""
            p = multiprocessing.Process(
                target=_adaptive_worker_subprocess,
                args=(
                    self.manager_url,
                    f"{self.worker_id}-sub{worker_num}",
                    self.worker_password,
                    self.processor.config_name,
                    self.processor.model_path,
                    gpu_id,
                    languages,
                    self.processor.batch_size,
                    self.s3_uploader.bucket if self.s3_uploader else None,
                    result_queue,
                    shared_stats,
                ),
                daemon=True
            )
            p.start()
            logger.info(f"Spawned worker subprocess {worker_num} (PID: {p.pid})")
            return p

        try:
            # Start with 1 worker
            # NOTE: We don't spawn here - this process IS the first worker
            # We'll spawn ADDITIONAL workers as subprocesses

            # This process becomes worker 0 and processes jobs directly
            logger.info("Main process will handle jobs as worker-0")

            while True:
                # Check for completed workers and collect results
                dead_workers = []
                for wid, proc in active_workers.items():
                    if not proc.is_alive():
                        dead_workers.append(wid)
                        exit_code = proc.exitcode
                        if exit_code != 0:
                            logger.warning(f"Worker {wid} died with exit code {exit_code}")
                        else:
                            logger.info(f"Worker {wid} exited normally")

                for wid in dead_workers:
                    del active_workers[wid]

                # Collect results from queue
                while not result_queue.empty():
                    try:
                        result = result_queue.get_nowait()
                        if result.get("success"):
                            consecutive_successes += 1
                            if shared_stats:
                                shared_stats.increment_completed()
                        else:
                            consecutive_successes = 0
                            if shared_stats:
                                shared_stats.increment_failed()
                            if result.get("oom"):
                                if shared_stats:
                                    shared_stats.increment_oom()
                                # OOM - reduce target
                                current_target = max(1, current_target - 1)
                                logger.warning(f"OOM detected, reducing target to {current_target}")
                    except:
                        break

                # Check if we should adjust concurrency
                now = time.time()
                if now - last_ramp_check >= ramp_check_interval:
                    safe_count = get_safe_worker_count()

                    # Consider ramping up if successful and memory allows
                    if consecutive_successes >= 3 and safe_count > current_target:
                        old_target = current_target
                        current_target = min(current_target + 1, safe_count, max_concurrency)
                        if current_target > old_target:
                            logger.info(f"Ramping up: {old_target} -> {current_target} workers (memory allows {safe_count})")
                            consecutive_successes = 0

                    last_ramp_check = now

                # Update stats more frequently (every ramp check)
                if shared_stats:
                    stats = memory_monitor.get_memory_stats()
                    if stats:
                        shared_stats.update_gpu_stats(gpu_id, {
                            "current_concurrency": current_target,
                            "active_jobs": len(active_workers) + 1,  # +1 for main process
                            "max_concurrency": max_concurrency,
                            "memory_used_mb": stats.used_mb,
                            "memory_total_mb": stats.total_mb,
                            "memory_utilization_pct": stats.utilization_pct,
                            "jobs_completed": shared_stats._global.get("total_completed", 0),
                        })

                # Spawn more workers if needed (current_target - 1 because main process is worker 0)
                workers_needed = current_target - 1 - len(active_workers)
                if workers_needed > 0:
                    # Check memory before spawning
                    stats = memory_monitor.get_memory_stats()
                    if stats and stats.free_mb > (estimated_worker_vram + safety_buffer_mb):
                        worker_counter += 1
                        proc = spawn_worker(worker_counter)
                        active_workers[worker_counter] = proc
                    else:
                        logger.debug(f"Skipping worker spawn - insufficient memory ({stats.free_mb:.0f}MB free)")

                # Main process handles a job (acts as worker 0)
                job = self.request_job(languages=languages)

                if job:
                    job_id = job["job_id"]
                    logger.info(f"[Main] Processing job #{job_id}")
                    try:
                        success = self._process_job_safe(job)
                        if success:
                            consecutive_successes += 1
                            if shared_stats:
                                shared_stats.increment_completed()
                        else:
                            consecutive_successes = 0
                            if shared_stats:
                                shared_stats.increment_failed()
                    except torch.cuda.OutOfMemoryError:
                        logger.error(f"[Main] OOM on job #{job_id}")
                        consecutive_successes = 0
                        current_target = max(1, current_target - 1)
                        if shared_stats:
                            shared_stats.increment_failed()
                            shared_stats.increment_oom()
                        # Force cleanup
                        memory_monitor.force_memory_cleanup()
                        time.sleep(5)
                    except Exception as e:
                        logger.error(f"[Main] Error on job #{job_id}: {e}")
                        consecutive_successes = 0
                        if shared_stats:
                            shared_stats.increment_failed()
                else:
                    # No jobs available
                    if not active_workers:
                        logger.info(f"No jobs available, waiting {poll_interval}s...")
                        time.sleep(poll_interval)
                    else:
                        # Have subprocess workers, they'll pick up jobs
                        time.sleep(2)

        except KeyboardInterrupt:
            logger.info("\nShutting down adaptive workers...")

        finally:
            # Terminate all subprocess workers
            for wid, proc in active_workers.items():
                if proc.is_alive():
                    logger.info(f"Terminating worker {wid}...")
                    proc.terminate()
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.kill()

            logger.info("Adaptive worker stopped")

    def _process_job_safe(self, job: Dict) -> bool:
        """
        Process a job with OOM-safe error handling.

        This is a simplified version of process_job designed for concurrent execution.
        It doesn't do prefetching (which doesn't make sense with concurrent jobs).
        """
        job_id = job["job_id"]
        episode_url = job["episode_url"]
        language = job.get("language")

        logger.info(f"[Job #{job_id}] Starting processing...")

        # Update status to processing
        self.update_job_status(job_id, "processing")

        temp_dir = tempfile.mkdtemp()
        upload_thread = None

        try:
            # Download audio
            audio_path = self.download_audio(episode_url, temp_dir)
            original_audio_path = audio_path

            # Start S3 upload in background if configured
            if self.s3_uploader:
                file_hash = self.processor.compute_file_hashes(original_audio_path)["sha256"]
                upload_thread = self.s3_uploader.upload_file_threaded(
                    original_audio_path, file_hash
                )
                logger.info(f"[Job #{job_id}] S3 upload started in background")

            # Process audio (GPU work happens here)
            results = self.processor.diarize_audio(
                original_audio_path,
                episode_url=episode_url,
                language=language,
            )

            # Submit results
            success = self.submit_results(job_id, results)

            # Wait for S3 upload to complete
            if upload_thread:
                upload_thread.join(timeout=120)  # 2 minute timeout for upload

            logger.info(f"[Job #{job_id}] Completed successfully")
            return success

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"[Job #{job_id}] OOM error: {e}")
            self.report_failure(job_id, f"OOM: {str(e)}")
            # Re-raise to let scheduler handle OOM
            raise

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"[Job #{job_id}] CUDA OOM error: {e}")
                self.report_failure(job_id, f"CUDA OOM: {str(e)}")
                raise torch.cuda.OutOfMemoryError(str(e))
            else:
                logger.error(f"[Job #{job_id}] Runtime error: {e}", exc_info=True)
                self.report_failure(job_id, str(e))
                return False

        except Exception as e:
            logger.error(f"[Job #{job_id}] Error: {e}", exc_info=True)
            self.report_failure(job_id, str(e))
            return False

        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def _adaptive_worker_subprocess(
    manager_url: str,
    worker_id: str,
    worker_password: Optional[str],
    config: str,
    model_path: Optional[str],
    gpu_id: int,
    languages: str,
    batch_size: int,
    s3_bucket: Optional[str],
    result_queue: multiprocessing.Queue,
    shared_stats: Optional['SharedWorkerStats'],
):
    """
    Worker subprocess that loads its own models and processes jobs continuously.

    This function runs in a separate process spawned by run_loop_adaptive.
    Each subprocess:
    1. Loads its own copy of the ML models
    2. Continuously requests and processes jobs
    3. Reports results back via the result_queue
    """
    import signal

    # Setup logging for this subprocess
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [{worker_id}] - %(levelname)s - %(message)s",
    )
    sub_logger = logging.getLogger(__name__)

    sub_logger.info(f"Subprocess worker starting on GPU {gpu_id}...")

    # Handle graceful shutdown
    shutdown_requested = False

    def handle_signal(sig, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        # Create worker instance (this loads models - takes time and VRAM)
        worker = PodcastPileWorker(
            manager_url=manager_url,
            worker_id=worker_id,
            worker_password=worker_password,
            config=config,
            model_path=model_path,
            gpu_id=gpu_id,
            languages=languages,
            batch_size=batch_size,
            s3_config=None,  # TODO: Pass S3 config if needed
        )

        sub_logger.info("Loading models...")
        worker.load_models()
        sub_logger.info("Models loaded, starting job loop")

        # Process jobs until shutdown
        while not shutdown_requested:
            job = worker.request_job(languages=languages)

            if job:
                job_id = job["job_id"]
                sub_logger.info(f"Processing job #{job_id}")

                try:
                    success = worker._process_job_safe(job)
                    result_queue.put({"success": success, "job_id": job_id})
                except torch.cuda.OutOfMemoryError as e:
                    sub_logger.error(f"OOM on job #{job_id}: {e}")
                    result_queue.put({"success": False, "job_id": job_id, "oom": True})
                    # Exit subprocess on OOM - let parent decide whether to respawn
                    break
                except Exception as e:
                    sub_logger.error(f"Error on job #{job_id}: {e}")
                    result_queue.put({"success": False, "job_id": job_id})
            else:
                # No jobs available, wait briefly
                time.sleep(5)

    except Exception as e:
        sub_logger.error(f"Subprocess worker failed: {e}", exc_info=True)
        result_queue.put({"success": False, "error": str(e)})

    sub_logger.info("Subprocess worker exiting")


