"""Utility functions for GRPO training."""

import gc
from typing import Tuple, Optional

import jax
import jax.numpy as jnp


class MemoryMonitor:
    """Monitor TPU memory usage."""
    
    @staticmethod
    def get_usage() -> Tuple[float, float]:
        """Get current memory usage.
        
        Returns:
            Tuple of (used_bytes, limit_bytes)
        """
        try:
            stats = [d.memory_stats() for d in jax.devices() if d.memory_stats()]
            if stats:
                used = sum(s['bytes_in_use'] for s in stats)
                limit = sum(s['bytes_limit'] for s in stats)
                return used, limit
        except Exception:
            pass
        return 0, 0
    
    @staticmethod
    def get_usage_gb() -> Tuple[float, float]:
        """Get memory usage in gigabytes.
        
        Returns:
            Tuple of (used_gb, limit_gb)
        """
        used, limit = MemoryMonitor.get_usage()
        return used / 1e9, limit / 1e9
    
    @staticmethod
    def get_percent() -> float:
        """Get memory usage percentage.
        
        Returns:
            Usage percentage (0-100)
        """
        used, limit = MemoryMonitor.get_usage()
        if limit > 0:
            return 100 * used / limit
        return 0.0
    
    @staticmethod
    def print_summary():
        """Print memory usage summary."""
        used, limit = MemoryMonitor.get_usage()
        if limit > 0:
            print(f"  TPU Memory: {used/1e9:.2f}GB / {limit/1e9:.2f}GB ({100*used/limit:.1f}%)")
        else:
            print("  TPU Memory: stats unavailable")
    
    @staticmethod
    def check_available(required_gb: float = 20.0) -> bool:
        """Check if sufficient memory is available.
        
        Args:
            required_gb: Required free memory in gigabytes
            
        Returns:
            True if sufficient memory available
        """
        used, limit = MemoryMonitor.get_usage()
        available = (limit - used) / 1e9
        return available >= required_gb


def clear_memory():
    """Clear Python garbage and JAX caches."""
    gc.collect()
    jax.clear_caches()


def get_cpu_device() -> Optional[jax.Device]:
    """Get CPU device for offloading.
    
    Returns:
        CPU device or None if unavailable
    """
    try:
        return jax.devices('cpu')[0]
    except Exception:
        return None


def create_mesh(shape: Tuple[int, ...], axes: Tuple[str, ...]) -> jax.sharding.Mesh:
    """Create JAX mesh for distributed training.
    
    Args:
        shape: Mesh shape (e.g., (2, 4))
        axes: Axis names (e.g., ("fsdp", "tp"))
        
    Returns:
        JAX Mesh object
    """
    return jax.make_mesh(
        shape,
        axes,
        axis_types=(jax.sharding.AxisType.Auto,) * len(shape)
    )


def convert_to_bf16(pytree):
    """Convert pytree arrays to bfloat16.
    
    Args:
        pytree: JAX pytree with arrays
        
    Returns:
        Pytree with bf16 arrays
    """
    return jax.tree.map(lambda x: x.astype(jnp.bfloat16), pytree)


def print_device_info():
    """Print JAX device information."""
    devices = jax.devices()
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {len(devices)}")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d.platform}: {d.device_kind}")
