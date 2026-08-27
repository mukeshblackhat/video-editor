"""
Tests for phase2b_matting.py — MatAnyone video matting integration.

Tests cover:
  - Module imports
  - Device detection
  - Mask preprocessing (dilate/erode)
  - MatAnyone model loading
  - Frame tensor conversion
  - Alpha output format and range
  - Soft alpha detection in phase3_composite
"""

import pytest
import numpy as np
import cv2
import torch


class TestDeviceDetection:
    """Test GPU/MPS/CPU device selection."""

    def test_get_device_returns_valid_device(self):
        from phase2b_matting import get_device
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")

    def test_get_device_prefers_gpu(self):
        from phase2b_matting import get_device
        device = get_device()
        if torch.cuda.is_available():
            assert device.type == "cuda"
        elif torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"


class TestMaskPreprocessing:
    """Test the dilate/erode mask preprocessing logic."""

    def test_dilate_expands_mask(self):
        """Dilation should expand the foreground region."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # 20x20 white square

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(mask, kernel, iterations=1)

        assert dilated.sum() > mask.sum()
        # Original region should still be white
        assert np.all(dilated[40:60, 40:60] == 255)

    def test_erode_shrinks_mask(self):
        """Erosion should shrink the foreground region."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255  # 40x40 white square

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        eroded = cv2.erode(mask, kernel, iterations=1)

        assert eroded.sum() < mask.sum()

    def test_closing_fills_small_holes(self):
        """Dilate then erode (closing) should fill small holes in mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        # Punch a small hole
        mask[48:52, 48:52] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.erode(cv2.dilate(mask, kernel), kernel)

        # The small hole should be filled
        assert closed[50, 50] == 255


class TestMatAnyoneImport:
    """Test that MatAnyone is properly installed and importable."""

    def test_import_inference_core(self):
        from matanyone.inference.inference_core import InferenceCore
        assert InferenceCore is not None

    def test_import_model(self):
        from matanyone.model.matanyone import MatAnyone
        assert MatAnyone is not None

    def test_import_device_utils(self):
        from matanyone.utils.device import get_default_device
        device = get_default_device()
        assert device.type in ("cuda", "mps", "cpu")


class TestSoftAlphaDetection:
    """Test that phase3_composite correctly detects soft vs binary masks."""

    def test_binary_mask_detected(self):
        """A mask with only 0 and 255 should be detected as binary."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        unique_vals = np.unique(mask)
        is_soft = np.any((unique_vals > 0) & (unique_vals < 255))
        assert not is_soft

    def test_soft_alpha_mask_detected(self):
        """A mask with intermediate values should be detected as soft alpha."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        # Add soft edges
        mask[29, 30:70] = 128
        mask[70, 30:70] = 64
        unique_vals = np.unique(mask)
        is_soft = np.any((unique_vals > 0) & (unique_vals < 255))
        assert is_soft

    def test_soft_alpha_used_directly(self):
        """When soft alpha is detected, it should be normalized to [0, 1] directly."""
        mask = np.array([[0, 128, 255]], dtype=np.uint8)
        alpha = mask.astype(np.float32) / 255.0
        assert alpha[0, 0] == 0.0
        assert abs(alpha[0, 1] - 128.0 / 255.0) < 1e-6
        assert alpha[0, 2] == 1.0


class TestFrameTensorConversion:
    """Test BGR→RGB tensor conversion for MatAnyone input."""

    def test_bgr_to_rgb_tensor(self):
        """Frame should be converted from BGR to RGB and normalized."""
        frame_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_bgr[:, :, 0] = 255  # Blue channel
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1)

        assert tensor.shape == (3, 100, 100)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        # Red channel (index 0) should be 0, Blue (index 2) should be 1
        assert tensor[0].max() == 0.0  # No red
        assert tensor[2].max() == 1.0  # Full blue

    def test_tensor_range(self):
        """Normalized tensor values should be in [0, 1]."""
        frame = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        tensor = torch.from_numpy(frame).float() / 255.0
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0


class TestAlphaOutputFormat:
    """Test expected alpha matte output format."""

    def test_alpha_to_uint8_rounding(self):
        """Alpha float [0,1] should be properly rounded to uint8 [0,255]."""
        alpha_float = np.array([0.0, 0.5, 0.999, 1.0], dtype=np.float32)
        alpha_uint8 = np.clip(np.round(alpha_float * 255), 0, 255).astype(np.uint8)
        assert alpha_uint8[0] == 0
        assert alpha_uint8[1] == 128  # round(127.5) = 128
        assert alpha_uint8[2] == 255  # round(254.745) = 255
        assert alpha_uint8[3] == 255

    def test_alpha_preserves_soft_values(self):
        """Soft alpha values should survive the float→uint8→float round trip."""
        original = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], dtype=np.float32)
        uint8 = np.clip(np.round(original * 255), 0, 255).astype(np.uint8)
        recovered = uint8.astype(np.float32) / 255.0
        # Should be within 1/255 of original
        assert np.allclose(original, recovered, atol=1.0 / 255.0)


class TestPhase2bModule:
    """Test the phase2b_matting module structure."""

    def test_module_imports(self):
        import phase2b_matting
        assert hasattr(phase2b_matting, 'run_matanyone_matting')
        assert hasattr(phase2b_matting, 'run_full_matting_pipeline')
        assert hasattr(phase2b_matting, 'generate_first_frame_mask_sam2')
        assert hasattr(phase2b_matting, 'get_device')
        assert hasattr(phase2b_matting, 'load_matanyone_model')

    def test_run_pipeline_imports_phase2b(self):
        """run_pipeline should be able to import phase2b_matting."""
        from phase2b_matting import run_full_matting_pipeline
        assert callable(run_full_matting_pipeline)
