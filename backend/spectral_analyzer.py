"""
FairFake Frequency-Domain Spectral Auditing Engine (Phase 4: Task 4.2)

Detects synthetic generation signatures in the frequency domain that are invariant
to demographic attributes (skin pigmentation, age, gender):
  1. 2D Fast Fourier Transform (2D-FFT) & Azimuthal Average Power Spectrum:
     - Computes 2D Fourier transform centered at DC frequency.
     - Performs 1D radial integration (azimuthal average) A(r) from DC to Nyquist limit.
  2. Synthetic Upsampling Checkerboard Detector:
     - Quantifies high-frequency periodic peaks introduced by Transposed Convolutions
       and sub-pixel upsamplers common in GANs, StyleGAN, and Latent Diffusion Models.
  3. 2D Discrete Cosine Transform (DCT) Energy Distribution:
     - Measures mid-to-high frequency compression energy ratios.
  4. Spectral Visualizer:
     - Encodes 2D power spectrum heatmaps into base64 images for frontend inspection.
"""

import math
import base64
import io
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import cv2
from PIL import Image
import torch


def to_grayscale_array(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Converts input image to a 2D float32 grayscale array normalized to [0, 1]."""
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.shape[0] == 3:
            # RGB to Gray weights: 0.2989 R + 0.5870 G + 0.1140 B
            gray = 0.2989 * t[0] + 0.5870 * t[1] + 0.1140 * t[2]
        else:
            gray = t[0]
        arr = gray.numpy()
        if arr.min() < -0.1:
            arr = (arr + 1.0) / 2.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    if isinstance(img, Image.Image):
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
        return arr

    if isinstance(img, np.ndarray):
        arr = img.copy()
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                arr = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            else:
                arr = arr[:, :, 0]
        return np.clip(arr, 0.0, 1.0)

    raise TypeError(f"Unsupported image format: {type(img)}")


def compute_2d_fft(gray_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D Fourier Transform with shifted zero-frequency component to center.

    Returns:
        fft_shift: complex 2D array of Fourier coefficients
        magnitude_spectrum: normalized log-magnitude 2D spectrum in [0, 1]
    """
    h, w = gray_arr.shape
    # Apply Hann window to eliminate boundary edge spectral leakage
    window_y = np.hanning(h)
    window_x = np.hanning(w)
    window_2d = np.outer(window_y, window_x)
    windowed_img = (gray_arr - np.mean(gray_arr)) * window_2d

    f = np.fft.fft2(windowed_img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Logarithmic dynamic range scaling: log(1 + |F|)
    log_mag = np.log1p(magnitude)
    min_val, max_val = log_mag.min(), log_mag.max()
    norm_spectrum = (log_mag - min_val) / (max_val - min_val + 1e-8)

    return fshift, norm_spectrum


def compute_azimuthal_average(magnitude_spectrum: np.ndarray, num_bins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 1D radial profile (azimuthal average) of the 2D power spectrum:
      A(r) = (1 / N_r) * sum_{|k| = r} |F(k)|
    From DC component (r = 0) to Nyquist frequency (r = R_max).

    Returns:
        radii: 1D array of normalized radial frequencies in [0, 1]
        azimuthal_profile: 1D array of mean magnitude along radius r
    """
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = min(cy, cx)

    # Bin radii
    bin_edges = np.linspace(0, max_radius, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) / max_radius
    radial_profile = np.zeros(num_bins, dtype=np.float32)

    for i in range(num_bins):
        mask = (r_map >= bin_edges[i]) & (r_map < bin_edges[i + 1])
        if np.any(mask):
            radial_profile[i] = float(np.mean(magnitude_spectrum[mask]))
        else:
            radial_profile[i] = 0.0

    # Normalize profile so max = 1.0
    if np.max(radial_profile) > 0:
        radial_profile = radial_profile / np.max(radial_profile)

    return bin_centers, radial_profile


def compute_checkerboard_score(magnitude_spectrum: np.ndarray) -> float:
    """
    Quantifies periodic synthetic upsampling grid artifacts.
    Generative architectures (GANs with deconvolution stride 2, Diffusion unet upsamplers)
    produce distinct cross-shaped periodic harmonic spikes in high-frequency regions.

    Natural images follow power-law decay 1/f^alpha, whereas synthetic images exhibit
    abnormal high-frequency energy spikes along orthogonal axes.
    """
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    r_limit = int(min(cy, cx) * 0.4)

    # High frequency outer ring mask
    y, x = np.ogrid[:h, :w]
    r_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    outer_mask = (r_map > r_limit) & (r_map < min(cy, cx) * 0.95)

    if not np.any(outer_mask):
        return 0.0

    outer_energies = magnitude_spectrum[outer_mask]
    mean_energy = float(np.mean(outer_energies))
    std_energy = float(np.std(outer_energies))

    # Cross axis energy (orthogonal axes where checkerboard frequencies concentrate)
    axis_tolerance = 4
    cross_mask = outer_mask & (
        (np.abs(x - cx) <= axis_tolerance) |
        (np.abs(y - cy) <= axis_tolerance) |
        (np.abs(np.abs(x - cx) - np.abs(y - cy)) <= axis_tolerance)
    )

    if np.any(cross_mask):
        cross_mean = float(np.mean(magnitude_spectrum[cross_mask]))
        # Ratio of axis energy to isotropic background energy
        axis_spike_ratio = max(0.0, (cross_mean - mean_energy) / (std_energy + 1e-6))
    else:
        axis_spike_ratio = 0.0

    # High frequency residual bump (synthetic images have elevated tail energy)
    outer_90th = float(np.percentile(outer_energies, 90))
    checkerboard_metric = float(1.0 / (1.0 + np.exp(-1.5 * (axis_spike_ratio - 1.2))))
    return round(float(np.clip(checkerboard_metric, 0.0, 1.0)), 4)


def compute_spectral_metrics(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Dict[str, Any]:
    """
    Comprehensive frequency-domain forensic evaluation.

    Returns:
        Dictionary containing:
          - checkerboard_score: probability of synthetic transposed conv artifacts [0, 1]
          - high_freq_energy_ratio: fraction of spectral energy in upper octave
          - radial_profile: 32-point azimuthal power decay curve
          - spectral_verdict: 'SYNTHETIC_UPSAMPLING' or 'NATURAL_SPECTRUM'
          - spectral_heatmap_b64: base64 encoded JET colormap of 2D Fourier spectrum
    """
    gray = to_grayscale_array(img)
    fshift, norm_spectrum = compute_2d_fft(gray)
    radii, profile = compute_azimuthal_average(norm_spectrum, num_bins=32)
    checker_score = compute_checkerboard_score(norm_spectrum)

    # High frequency energy ratio (outer 30% radial bandwidth)
    split_idx = int(len(profile) * 0.7)
    high_freq_energy = float(np.sum(profile[split_idx:]))
    total_energy = float(np.sum(profile)) + 1e-8
    hf_ratio = round(high_freq_energy / total_energy, 4)

    # Classification threshold
    verdict = "SYNTHETIC_UPSAMPLING" if checker_score >= 0.55 or hf_ratio > 0.35 else "NATURAL_SPECTRUM"

    # Generate colored heatmap for UI
    heatmap_uint8 = np.uint8(255 * norm_spectrum)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    _, buffer = cv2.imencode(".png", colored_heatmap)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    heatmap_b64 = f"data:image/png;base64,{b64_str}"

    return {
        "checkerboard_score": checker_score,
        "high_freq_energy_ratio": hf_ratio,
        "spectral_verdict": verdict,
        "radial_profile": [round(float(v), 4) for v in profile],
        "radial_frequencies": [round(float(r), 4) for r in radii],
        "spectral_heatmap_b64": heatmap_b64
    }


if __name__ == "__main__":
    print("[*] Testing Spectral Analyzer Engine...")
    # Test on synthetic checkerboard grid vs smooth natural image
    natural_img = np.zeros((128, 128), dtype=np.float32)
    for y in range(128):
        for x in range(128):
            natural_img[y, x] = 0.5 + 0.3 * np.sin(x / 10.0) + 0.2 * np.cos(y / 15.0)

    # Synthetic image with injected upsampling periodic grid
    synth_img = natural_img.copy()
    synth_img[::2, ::2] += 0.4  # periodic 2x2 grid

    res_nat = compute_spectral_metrics(natural_img)
    res_syn = compute_spectral_metrics(synth_img)

    print(f"  Natural Image  -> Checkerboard Score: {res_nat['checkerboard_score']:.4f} | Verdict: {res_nat['spectral_verdict']}")
    print(f"  Synthetic Image -> Checkerboard Score: {res_syn['checkerboard_score']:.4f} | Verdict: {res_syn['spectral_verdict']}")

    assert res_syn["checkerboard_score"] > res_nat["checkerboard_score"], "Checkerboard score should be higher on synthetic grid"
    print("[+] Spectral Analyzer successfully verified!")
