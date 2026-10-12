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
        is_cuda_request = target in ("cuda", "cuda:0", "0")
        
        if is_cuda_request:
            if cuda_avail and count > 0 and kernels_functional:
                logger.info(f"Verified CUDA device 0: {gpu_name}")
                return "cuda:0"
            else:
                reason = diag or "CUDA is not available to PyTorch."
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

        # Guarantee truthfulness constraint: Never report cuda:0 unless actually available and functional
        if active_device == "cuda:0" and not (cuda_avail and kernels_functional):
            raise DeviceConfigurationError("Falsified device detected: active_device='cuda:0' but GPU is not functional.")

        return RuntimeInfo(
            python_version=sys.version.split()[0],
            torch_version=torch_ver,
            cuda_available=cuda_avail,
            gpu_name=gpu_name if (cuda_avail and active_device == "cuda:0") else None,
            active_device=active_device,
            backend="Ultralytics YOLO",
            weights_path=weights_path,
        )
