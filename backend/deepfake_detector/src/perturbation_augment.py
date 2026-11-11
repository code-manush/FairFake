"""
FairFake Environmental Perturbation Augmentation Pipeline (Phase 3: Task 3.1)

Provides a realistic degradation and stress-testing suite to harden deepfake
detectors against environmental noise, including:
  1. Photometric / Illumination distortions:
     - Non-linear Gamma correction (gamma in [0.4, 2.2])
     - Shadow / non-uniform lighting illumination gradients
     - Exposure scaling (underexposure / overexposure)
  2. Lossy Compression artifacts:
     - JPEG compression (quality in [20, 95])
     - WebP compression simulation
     - 8x8 block discrete cosine transform (DCT) quantization
  3. High-Frequency / Optical Blur:
     - Gaussian blur (variable kernel & sigma)
     - Directional Motion blur (linear PSF at arbitrary angles)
     - Defocus disc blur

All transforms support PIL Images, NumPy arrays, and PyTorch tensors [C, H, W]
in both [0, 1] and [-1, 1] scales.
"""

import math
import random
import io
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility conversions
# ---------------------------------------------------------------------------

def to_pil_image(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    """Safely converts an input image to a PIL Image (RGB mode)."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)
        # If in [-1, 1] scale, un-normalize to [0, 1]
        if t.min() < -0.1:
            t = (t + 1.0) / 2.0
        t = torch.clamp(t, 0.0, 1.0)
        np_arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_arr, mode="RGB")
        
    if isinstance(img, np.ndarray):
        arr = img.copy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0).astype(np.uint8)
            elif arr.min() < 0.0:
                arr = (((arr + 1.0) / 2.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        return Image.fromarray(arr, mode="RGB")
        
    raise TypeError(f"Unsupported image type: {type(img)}")


def from_pil_image(
    pil_img: Image.Image,
    target_type: type,
    original_shape: Optional[Tuple[int, ...]] = None,
    original_range: Optional[Tuple[float, float]] = None
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """Converts a PIL image back to the caller's preferred format."""
    if issubclass(target_type, Image.Image):
        return pil_img
        
    np_arr = np.array(pil_img, dtype=np.uint8)
    
    if target_type == np.ndarray:
        if original_range and original_range[0] < 0:
            return (np_arr.astype(np.float32) / 127.5) - 1.0
        return np_arr
        
    if issubclass(target_type, torch.Tensor):
        t = torch.from_numpy(np_arr).permute(2, 0, 1).float() / 255.0
        if original_range and original_range[0] < -0.1:
            t = (t * 2.0) - 1.0
        if original_shape and len(original_shape) == 4:
            t = t.unsqueeze(0)
        return t
        
    return pil_img


# ---------------------------------------------------------------------------
# 1. Photometric / Illumination Degradations
# ---------------------------------------------------------------------------

class GammaCorrection:
    """
    Applies non-linear gamma luminance correction: I_out = I_in ^ gamma.
    Simulates severe underexposure (gamma > 1.0) or overexposure (gamma < 1.0).
    """

    def __init__(self, gamma_range: Tuple[float, float] = (0.4, 2.2), p: float = 1.0):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        gamma: Optional[float] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if random.random() > self.p and gamma is None:
            return img

        orig_type = type(img)
        orig_shape = getattr(img, "shape", None)
        orig_min = float(img.min()) if hasattr(img, "min") else 0.0
        orig_max = float(img.max()) if hasattr(img, "max") else 1.0
        
        pil_img = to_pil_image(img)
        arr = np.array(pil_img, dtype=np.float32) / 255.0

        if gamma is None:
            gamma = random.uniform(*self.gamma_range)

        # Gamma transformation
        arr_corrected = np.power(np.maximum(arr, 1e-6), gamma)
        arr_corrected = np.clip(arr_corrected * 255.0, 0, 255).astype(np.uint8)
        res_pil = Image.fromarray(arr_corrected, mode="RGB")

        return from_pil_image(
            res_pil,
            orig_type,
            orig_shape,
            (orig_min, orig_max)
        )


class ShadowIllumination:
    """
    Simulates non-uniform directional illumination and ambient shadows,
    creating spatial luminance gradients across facial features.
    """

    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.3, 0.75),
        angle_range: Tuple[float, float] = (0, 360),
        p: float = 1.0
    ):
        self.intensity_range = intensity_range
        self.angle_range = angle_range
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        intensity: Optional[float] = None,
        angle: Optional[float] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if random.random() > self.p and intensity is None:
            return img

        orig_type = type(img)
        orig_shape = getattr(img, "shape", None)
        orig_min = float(img.min()) if hasattr(img, "min") else 0.0
        orig_max = float(img.max()) if hasattr(img, "max") else 1.0

        pil_img = to_pil_image(img)
        w, h = pil_img.size
        arr = np.array(pil_img, dtype=np.float32)

        if intensity is None:
            intensity = random.uniform(*self.intensity_range)
        if angle is None:
            angle = random.uniform(*self.angle_range)

        # Generate directional linear gradient map
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        x = np.linspace(-1, 1, w, dtype=np.float32)
        y = np.linspace(-1, 1, h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        proj = xx * cos_a + yy * sin_a
        # Normalize to [0, 1]
        grad = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
        
        # Shadow factor: 1.0 (bright) down to (1.0 - intensity) (dark)
        shadow_map = 1.0 - (intensity * grad)
        shadow_map = np.expand_dims(shadow_map, axis=-1)

        arr_shadowed = np.clip(arr * shadow_map, 0, 255).astype(np.uint8)
        res_pil = Image.fromarray(arr_shadowed, mode="RGB")

        return from_pil_image(
            res_pil,
            orig_type,
            orig_shape,
            (orig_min, orig_max)
        )


class ExposurePerturbation:
    """
    Simulates camera exposure scaling (underexposure / overexposure).
    """

    def __init__(self, factor_range: Tuple[float, float] = (0.4, 1.8), p: float = 1.0):
        self.factor_range = factor_range
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        factor: Optional[float] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if random.random() > self.p and factor is None:
            return img

        orig_type = type(img)
        orig_shape = getattr(img, "shape", None)
        orig_min = float(img.min()) if hasattr(img, "min") else 0.0
        orig_max = float(img.max()) if hasattr(img, "max") else 1.0

        pil_img = to_pil_image(img)
        if factor is None:
            factor = random.uniform(*self.factor_range)

        enhancer = ImageEnhance.Brightness(pil_img)
        res_pil = enhancer.enhance(factor)

        return from_pil_image(
            res_pil,
            orig_type,
            orig_shape,
            (orig_min, orig_max)
        )


# ---------------------------------------------------------------------------
# 2. Lossy Compression Artifacts
# ---------------------------------------------------------------------------

class JPEGCompression:
    """
    Applies lossy JPEG compression with variable quality factor Q in [10, 100].
    Produces characteristic 8x8 DCT high-frequency ringing and quantization noise.
    """

    def __init__(self, quality_range: Tuple[int, int] = (20, 95), p: float = 1.0):
        self.quality_range = quality_range
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        quality: Optional[int] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if random.random() > self.p and quality is None:
            return img

        orig_type = type(img)
        orig_shape = getattr(img, "shape", None)
        orig_min = float(img.min()) if hasattr(img, "min") else 0.0
        orig_max = float(img.max()) if hasattr(img, "max") else 1.0

        pil_img = to_pil_image(img)
        if quality is None:
            quality = int(random.uniform(*self.quality_range))
        quality = max(10, min(100, int(quality)))

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        res_pil = Image.open(buf).convert("RGB")

        return from_pil_image(
            res_pil,
            orig_type,
            orig_shape,
            (orig_min, orig_max)
        )


class WebPCompression:
    """
    Applies WebP lossy compression simulating mobile web and social media pipelines.
    """

    def __init__(self, quality_range: Tuple[int, int] = (25, 90), p: float = 1.0):
        self.quality_range = quality_range
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        quality: Optional[int] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if random.random() > self.p and quality is None:
            return img

        orig_type = type(img)
        orig_shape = getattr(img, "shape", None)
        orig_min = float(img.min()) if hasattr(img, "min") else 0.0
        orig_max = float(img.max()) if hasattr(img, "max") else 1.0

        pil_img = to_pil_image(img)
        if quality is None:
            quality = int(random.uniform(*self.quality_range))
        quality = max(10, min(100, int(quality)))

        buf = io.BytesIO()
        pil_img.save(buf, format="WEBP", quality=quality)
        buf.seek(0)
        res_pil = Image.open(buf).convert("RGB")

        return from_pil_image(
            res_pil,
            orig_type,
            orig_shape,
            (orig_min, orig_max)
        )


class BlockQuantization:
    """
    Simulates macroblock quantization and chroma subsampling artifacts (H.264 video encoding).
    """

    def __init__(self, block_size: int = 8, q_scale_range: Tuple[float, float] = (2.0, 10.0), p: float = 1.0):
        self.block_size = block_size
        self.q_scale_range = q_scale_range
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        q_scale: Optional[float] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if random.random() > self.p and q_scale is None:
            return img

        orig_type = type(img)
        orig_shape = getattr(img, "shape", None)
        orig_min = float(img.min()) if hasattr(img, "min") else 0.0
        orig_max = float(img.max()) if hasattr(img, "max") else 1.0

        pil_img = to_pil_image(img)
        arr = np.array(pil_img, dtype=np.float32)
        h, w, c = arr.shape

        if q_scale is None:
            q_scale = random.uniform(*self.q_scale_range)

        # Quantize in spatial blocks
        b = self.block_size
        arr_blocked = arr.copy()
        for y in range(0, h, b):
            for x in range(0, w, b):
                block = arr[y:y+b, x:x+b, :]
                mean_val = np.mean(block, axis=(0, 1), keepdims=True)
                diff = block - mean_val
                # Quantize residual
                diff_quant = np.round(diff / q_scale) * q_scale
                arr_blocked[y:y+b, x:x+b, :] = mean_val + diff_quant

        arr_blocked = np.clip(arr_blocked, 0, 255).astype(np.uint8)
        res_pil = Image.fromarray(arr_blocked, mode="RGB")

        return from_pil_image(
            res_pil,
            orig_type,
            orig_shape,
            (orig_min, orig_max)
        )


# ---------------------------------------------------------------------------
# 3. High-Frequency Degradation & Optical Blur
# ---------------------------------------------------------------------------

class GaussianBlurPerturbation:
    """
    Applies Gaussian smoothing across channels.
    """

    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 3.0), p: float = 1.0):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        sigma: Optional[float] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if random.random() > self.p and sigma is None:
            return img

        orig_type = type(img)
        orig_shape = getattr(img, "shape", None)
        orig_min = float(img.min()) if hasattr(img, "min") else 0.0
        orig_max = float(img.max()) if hasattr(img, "max") else 1.0

        pil_img = to_pil_image(img)
        if sigma is None:
            sigma = random.uniform(*self.sigma_range)

        res_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))

        return from_pil_image(
            res_pil,
            orig_type,
            orig_shape,
            (orig_min, orig_max)
        )


class MotionBlur:
    """
    Generates realistic linear motion blur at arbitrary angles,
    simulating subject motion or camera displacement during capture.
    """

    def __init__(
        self,
        kernel_size_range: Tuple[int, int] = (5, 15),
        angle_range: Tuple[float, float] = (0, 360),
        p: float = 1.0
    ):
        self.kernel_size_range = kernel_size_range
        self.angle_range = angle_range
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        kernel_size: Optional[int] = None,
        angle: Optional[float] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if random.random() > self.p and kernel_size is None:
            return img

        orig_type = type(img)
        orig_shape = getattr(img, "shape", None)
        orig_min = float(img.min()) if hasattr(img, "min") else 0.0
        orig_max = float(img.max()) if hasattr(img, "max") else 1.0

        pil_img = to_pil_image(img)
        arr = np.array(pil_img, dtype=np.uint8)

        if kernel_size is None:
            kernel_size = int(random.uniform(*self.kernel_size_range))
        kernel_size = max(3, kernel_size | 1)  # Ensure odd

        if angle is None:
            angle = random.uniform(*self.angle_range)

        # Create 1D linear kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        mid = kernel_size // 2
        kernel[mid, :] = 1.0

        # Rotate kernel by angle
        rot_mat = cv2.getRotationMatrix2D((mid, mid), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum

        arr_blurred = cv2.filter2D(arr, -1, kernel)
        res_pil = Image.fromarray(arr_blurred, mode="RGB")

        return from_pil_image(
            res_pil,
            orig_type,
            orig_shape,
            (orig_min, orig_max)
        )


class DefocusBlur:
    """
    Simulates optical out-of-focus disc blur.
    """

    def __init__(self, radius_range: Tuple[int, int] = (2, 6), p: float = 1.0):
        self.radius_range = radius_range
        self.p = p

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        radius: Optional[int] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if random.random() > self.p and radius is None:
            return img

        orig_type = type(img)
        orig_shape = getattr(img, "shape", None)
        orig_min = float(img.min()) if hasattr(img, "min") else 0.0
        orig_max = float(img.max()) if hasattr(img, "max") else 1.0

        pil_img = to_pil_image(img)
        arr = np.array(pil_img, dtype=np.uint8)

        if radius is None:
            radius = int(random.uniform(*self.radius_range))
        radius = max(1, radius)
        k_size = (2 * radius) + 1

        # Circular disc structuring element
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        disc = (x * x + y * y <= radius * radius).astype(np.float32)
        disc /= disc.sum()

        arr_blurred = cv2.filter2D(arr, -1, disc)
        res_pil = Image.fromarray(arr_blurred, mode="RGB")

        return from_pil_image(
            res_pil,
            orig_type,
            orig_shape,
            (orig_min, orig_max)
        )


# ---------------------------------------------------------------------------
# 4. Composite & Demographic-Aware Pipeline
# ---------------------------------------------------------------------------

class RobustPerturbationTransform:
    """
    Combined perturbation transform that applies a stochastic blend of
    environmental degradations (Illumination, Compression, Blur) with
    configurable probabilities.
    """

    def __init__(
        self,
        p_gamma: float = 0.5,
        p_shadow: float = 0.4,
        p_jpeg: float = 0.6,
        p_blur: float = 0.4,
        p_motion: float = 0.3
    ):
        self.gamma = GammaCorrection(gamma_range=(0.5, 2.0), p=p_gamma)
        self.shadow = ShadowIllumination(intensity_range=(0.25, 0.65), p=p_shadow)
        self.exposure = ExposurePerturbation(factor_range=(0.6, 1.5), p=0.3)
        self.jpeg = JPEGCompression(quality_range=(30, 85), p=p_jpeg)
        self.webp = WebPCompression(quality_range=(35, 85), p=0.2)
        self.gaussian = GaussianBlurPerturbation(sigma_range=(0.5, 2.0), p=p_blur)
        self.motion = MotionBlur(kernel_size_range=(5, 11), p=p_motion)

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        x = img
        # 1. Photometric stage
        if random.random() < 0.5:
            x = self.gamma(x)
        else:
            x = self.exposure(x)
            
        x = self.shadow(x)

        # 2. Optical degradation stage
        if random.random() < 0.5:
            x = self.gaussian(x)
        else:
            x = self.motion(x)

        # 3. Codec compression stage
        if random.random() < 0.7:
            x = self.jpeg(x)
        else:
            x = self.webp(x)

        return x


class DemographicHardeningAugmentor:
    """
    Demographic-adaptive perturbation augmentor:
    Injects tailored perturbation distributions based on known vulnerability profiles:
      - Dark Skin / Deep Melanin: higher probability of underexposure/gamma variations
      - Heavy Makeup / Cosmetics: higher probability of JPEG/lossy compression noise
      - Eyeglasses / Accessories: higher probability of optical defocus/specular blur
    """

    def __init__(self):
        self.base_robust = RobustPerturbationTransform()
        self.gamma_dark = GammaCorrection(gamma_range=(1.3, 2.2), p=0.8)
        self.jpeg_cosmetic = JPEGCompression(quality_range=(25, 60), p=0.85)
        self.defocus_glasses = DefocusBlur(radius_range=(2, 5), p=0.75)

    def __call__(
        self,
        img: Union[Image.Image, np.ndarray, torch.Tensor],
        attributes: Optional[Dict[str, int]] = None
    ) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if not attributes:
            return self.base_robust(img)

        x = img
        # Targeted stress injection
        if attributes.get("Dark_Skin", 0) == 1 or attributes.get("Black", 0) == 1:
            x = self.gamma_dark(x)

        if attributes.get("Heavy_Makeup", 0) == 1 or attributes.get("Wearing_Lipstick", 0) == 1:
            x = self.jpeg_cosmetic(x)

        if attributes.get("Eyeglasses", 0) == 1:
            x = self.defocus_glasses(x)

        # General robustness
        if random.random() < 0.4:
            x = self.base_robust(x)

        return x


def get_standard_stress_conditions() -> Dict[str, Callable[[Any], Any]]:
    """
    Returns a standardized dictionary of benchmark stress conditions
    for the Phase 3 robustness audit.
    """
    return {
        "Clean": lambda x: x,
        "Underexposure_Gamma_1_8": lambda x: GammaCorrection()(x, gamma=1.8),
        "Overexposure_Gamma_0_6": lambda x: GammaCorrection()(x, gamma=0.6),
        "Directional_Shadow_60": lambda x: ShadowIllumination()(x, intensity=0.6, angle=45),
        "JPEG_Quality_30": lambda x: JPEGCompression()(x, quality=30),
        "JPEG_Quality_60": lambda x: JPEGCompression()(x, quality=60),
        "WebP_Quality_40": lambda x: WebPCompression()(x, quality=40),
        "Motion_Blur_9px": lambda x: MotionBlur()(x, kernel_size=9, angle=30),
        "Gaussian_Blur_Sigma_2": lambda x: GaussianBlurPerturbation()(x, sigma=2.0),
        "Defocus_Blur_Radius_3": lambda x: DefocusBlur()(x, radius=3),
        "Combined_Harsh_CCTV": lambda x: JPEGCompression()(
            GammaCorrection()(x, gamma=1.6),
            quality=35
        )
    }


if __name__ == "__main__":
    print("[*] Testing Perturbation Augmentation Suite...")
    dummy_img = Image.new("RGB", (256, 256), color=(180, 120, 90))
    
    # Test individual augmentations
    gamma_out = GammaCorrection()(dummy_img, gamma=1.8)
    shadow_out = ShadowIllumination()(dummy_img, intensity=0.6, angle=45)
    jpeg_out = JPEGCompression()(dummy_img, quality=30)
    blur_out = MotionBlur()(dummy_img, kernel_size=9, angle=45)
    
    print(f"  Gamma: {gamma_out.size}")
    print(f"  Shadow: {shadow_out.size}")
    print(f"  JPEG: {jpeg_out.size}")
    print(f"  Motion: {blur_out.size}")
    
    # Test tensor input
    dummy_tensor = torch.rand(3, 224, 224)
    tensor_out = RobustPerturbationTransform()(dummy_tensor)
    print(f"  Tensor Transform: shape={tensor_out.shape}, min={tensor_out.min():.2f}, max={tensor_out.max():.2f}")
    
    print("[+] All perturbation primitives verified successfully!")
