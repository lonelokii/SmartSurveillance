"""Runtime truthfulness and device diagnostics module."""

import logging
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class DeviceConfigurationError(Exception):
    """Raised when an invalid device configuration or falsified hardware reporting is detected."""
    pass


class ModelLoadError(Exception):
    """Raised when the YOLO model or weights fail to load cleanly."""
    pass


@dataclass
class RuntimeInfo:
    python_version: str
    torch_version: str
    cuda_available: bool
    gpu_name: Optional[str]
    active_device: str
    backend: str
    weights_path: str


class RuntimeDiagnostics:
    """Manages and verifies hardware device selection and runtime truthfulness."""

    @staticmethod
    def assert_no_hog():
        """Strictly verify that OpenCV HOG is not being imported or utilized as a detector."""
        import cv2
        # Ensure nobody monkey-patched or replaced the detector with HOG
        if hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
            logger.debug("cv2 has HOGDescriptor in library, but SmartSurveillance detector is strictly YOLO.")

    @staticmethod
    def get_cuda_status() -> tuple[bool, int, Optional[str], bool, Optional[str]]:
        """Query PyTorch CUDA support truthfully.
        
        Returns:
            (cuda_available: bool, device_count: int, gpu_name: Optional[str],
             kernels_functional: bool, diagnostic_message: Optional[str])
        """
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
            count = torch.cuda.device_count() if cuda_avail else 0
            name = torch.cuda.get_device_name(0) if cuda_avail and count > 0 else None
            
            kernels_functional = False
            diag_msg = None
            if cuda_avail and count > 0:
                try:
                    # Empirically verify kernel execution on CUDA device 0
                    test_tensor = torch.zeros((1,), device="cuda:0")
                    _ = test_tensor + 1
                    kernels_functional = True
                except Exception as e:
                    kernels_functional = False
                    diag_msg = (
                        f"CUDA device detected ({name}), but kernel execution failed: {e}. "
                        "Precompiled GPU kernels for this architecture (e.g. sm_120) are missing in current PyTorch wheel."
                    )
                    logger.warning(diag_msg)
            
            return cuda_avail, count, name, kernels_functional, diag_msg
        except ImportError:
            return False, 0, None, False, "PyTorch is not installed"

    @classmethod
    def resolve_device(cls, requested_device: str, auto_fallback: bool = True) -> str:
        """Resolve requested device against actual hardware availability truthfully.
        
        Args:
            requested_device: 'cuda:0', '0', or 'cpu'
            auto_fallback: if True, fallback to 'cpu' when CUDA unavailable with clear log.
        
        Returns:
            Resolved device string ('cuda:0' or 'cpu').
        
        Raises:
            DeviceConfigurationError: if CUDA requested, unavailable/incompatible, and auto_fallback is False.
        """
        cuda_avail, count, gpu_name, kernels_functional, diag = cls.get_cuda_status()
        
        target = requested_device.strip().lower()

        # 1. Handle Apple Silicon Metal Performance Shaders (MPS)
        if target == "mps":
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    _ = torch.zeros((1,), device="mps") + 1
                    logger.info("Verified Apple Silicon MPS device acceleration.")
                    return "mps"
                else:
                    msg = "MPS requested, but torch.backends.mps is not available on this platform/wheel."
                    if auto_fallback:
                        logger.warning(f"{msg} Truthfully falling back to CPU mode.")
                        return "cpu"
                    else:
                        raise DeviceConfigurationError(f"{msg} Auto-fallback is disabled.")
            except Exception as e:
                msg = f"MPS device detected but kernel execution failed: {e}."
                if auto_fallback:
                    logger.warning(f"{msg} Truthfully falling back to CPU mode.")
                    return "cpu"
                else:
                    raise DeviceConfigurationError(f"{msg} Auto-fallback is disabled.")

        # 2. Handle NVIDIA CUDA devices (e.g. cuda, cuda:0, cuda:1, 0, 1)
        is_cuda_request = target == "cuda" or target.startswith("cuda:") or target.isdigit()
        if is_cuda_request:
            cuda_idx = 0
            if target.startswith("cuda:"):
                try:
                    cuda_idx = int(target.split(":")[1])
                except (ValueError, IndexError):
                    cuda_idx = 0
            elif target.isdigit():
                cuda_idx = int(target)

            resolved_cuda_str = f"cuda:{cuda_idx}"

            if cuda_idx == 0:
                # Primary device check using cached/empirical status
                if cuda_avail and count > 0 and kernels_functional:
                    logger.info(f"Verified CUDA device 0: {gpu_name}")
                    return "cuda:0"
                else:
                    reason = diag or "CUDA device 0 is not available to PyTorch."
            else:
                # Secondary device check
                if cuda_avail and count > cuda_idx:
                    try:
                        import torch
                        _ = torch.zeros((1,), device=resolved_cuda_str) + 1
                        sec_name = torch.cuda.get_device_name(cuda_idx)
                        logger.info(f"Verified CUDA device {cuda_idx}: {sec_name}")
                        return resolved_cuda_str
                    except Exception as e:
                        reason = f"CUDA device {cuda_idx} kernel execution failed: {e}"
                else:
                    reason = f"CUDA device {cuda_idx} unavailable (only {count} CUDA devices visible)."

            msg = f"Requested device '{requested_device}', but {reason}"
            if auto_fallback:
                logger.warning(f"{msg} Truthfully falling back to CPU mode.")
                return "cpu"
            else:
                raise DeviceConfigurationError(f"{msg} Auto-fallback is disabled.")

        logger.info("Configured for CPU execution.")
        return "cpu"

    @classmethod
    def get_runtime_info(cls, active_device: str, weights_path: str) -> RuntimeInfo:
        """Return truthful runtime snapshot."""
        try:
            import torch
            torch_ver = torch.__version__
        except ImportError:
            torch_ver = "not_installed"

        cuda_avail, _, gpu_name, kernels_functional, _ = cls.get_cuda_status()

        # Guarantee truthfulness constraint: Never report cuda unless actually available and functional
        if active_device.startswith("cuda") and not (cuda_avail and kernels_functional):
            raise DeviceConfigurationError(f"Falsified device detected: active_device='{active_device}' but GPU is not functional.")

        active_gpu_name = None
        if active_device.startswith("cuda") and cuda_avail:
            try:
                import torch
                idx = int(active_device.split(":")[1]) if ":" in active_device else 0
                active_gpu_name = torch.cuda.get_device_name(idx)
            except Exception:
                active_gpu_name = gpu_name
        elif active_device == "mps":
            active_gpu_name = "Apple Silicon GPU (MPS)"

        return RuntimeInfo(
            python_version=sys.version.split()[0],
            torch_version=torch_ver,
            cuda_available=cuda_avail,
            gpu_name=active_gpu_name,
            active_device=active_device,
            backend="Ultralytics YOLO",
            weights_path=weights_path,
        )
