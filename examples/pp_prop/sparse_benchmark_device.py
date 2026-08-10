"""Backend selection and device memory accounting for the sparse benchmark.

The benchmark otherwise inherits whatever backend JAX binds, which makes a
requested accelerator indistinguishable from a silent fallback to the host.
``apply_device_selection`` pins the platform before any backend initializes and
``verify_device_selection`` refuses a run whose bound backend cannot satisfy the
request, so an absent accelerator fails loudly instead of reporting host numbers
under an accelerator heading.

.. code-block:: python

    >>> from sparse_benchmark_device import verify_device_selection
    >>> verify_device_selection('auto', 'cpu')
    >>> verify_device_selection('gpu', 'cpu')
    Traceback (most recent call last):
    RuntimeError: requested device gpu, bound backend is cpu
"""

from __future__ import annotations

import os
from typing import Literal, MutableMapping


DeviceSelection = Literal["auto", "cpu", "gpu"]

DEVICE_SELECTIONS: tuple[str, ...] = ("auto", "cpu", "gpu")

PLATFORM_VARIABLE = "JAX_PLATFORMS"

_GPU_PLATFORMS = frozenset({"gpu", "cuda", "rocm"})


def apply_device_selection(
    device: DeviceSelection, environment: MutableMapping[str, str] | None = None
) -> None:
    """Pin the JAX platform for this process before a backend initializes.

    Only ``"cpu"`` pins a platform. ``"gpu"`` states a requirement rather than a
    vendor, leaving JAX to choose among the accelerator plugins it has, and
    ``"auto"`` leaves any externally supplied setting untouched.

    Parameters
    ----------
    device : {"auto", "cpu", "gpu"}
        Requested backend.
    environment : mutable mapping of str to str, optional
        Mapping receiving the platform variable. The process environment is
        used when omitted.
    """
    if device != "cpu":
        return
    target = os.environ if environment is None else environment
    target[PLATFORM_VARIABLE] = "cpu"


def verify_device_selection(device: DeviceSelection, platform: str) -> None:
    """Raise when the bound backend cannot satisfy the requested device.

    Parameters
    ----------
    device : {"auto", "cpu", "gpu"}
        Requested backend.
    platform : str
        Platform reported by the device JAX actually bound.

    Raises
    ------
    RuntimeError
        If an accelerator was requested and the bound platform is not one.
    """
    if device == "gpu" and platform not in _GPU_PLATFORMS:
        raise RuntimeError(f"requested device gpu, bound backend is {platform}")


def device_memory_peak_bytes(device: object) -> int | None:
    """Return the allocator peak for ``device`` in bytes.

    The value is the peak live allocation reported by the XLA allocator, not the
    size of any preallocated pool. Backends that expose no statistics, the host
    backend among them, yield ``None``.

    Parameters
    ----------
    device : object
        Device whose allocator statistics are read.

    Returns
    -------
    int or None
        Peak bytes in use, or ``None`` when the backend reports nothing.
    """
    statistics = getattr(device, "memory_stats", None)
    if statistics is None:
        return None
    measured = statistics()
    if not isinstance(measured, dict):
        return None
    peak = measured.get("peak_bytes_in_use")
    return int(peak) if isinstance(peak, (int, float)) else None
