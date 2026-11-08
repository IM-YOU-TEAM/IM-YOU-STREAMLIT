"""canonical100_v1 input representation used by the 100 Hz shank/thigh BioMAT checkpoint.

Reusable by biomat_app.py and later phone/IMU runners (e.g. test_phone_imu.py).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

HZ = 100
LENGTH = 250
SITES = ['shank', 'thigh']
TARGETS = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
CLASSES = ['walking', 'ramp_ascent', 'ramp_descent', 'stair_ascent', 'stair_descent']
PREPROCESS = (
    'canonical100_v1: 20Hz lowpass; 250 samples; median acceleration norm; '
    'mean-acceleration + transverse-gyro PCA basis; peak-projection sign'
)


def is_canonical100_meta(meta) -> bool:
    """True when checkpoint metadata describes the canonical100_v1 contract."""
    if not meta:
        return False
    preprocess = str(meta.get('preprocess', '')).lower()
    if 'canonical100' in preprocess:
        return True
    return False


def as_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def lowpass_windows(x, cutoff_hz: float = 20.0, fs: float = HZ, order: int = 4) -> np.ndarray:
    """Fourth-order Butterworth filtfilt along time for each window/channel."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f'Expected [N, T, C] windows, got shape {x.shape}')
    sos = butter(order, cutoff_hz, fs=fs, output='sos')
    return sosfiltfilt(sos, x, axis=1)


def canonicalize(x):
    """Signal-derived rotation / acceleration-scale normalization per window.

    Frame is NOT an anatomical calibration. Sign may change between windows.
    Expects finite float arrays shaped [N, 250, 12] with shank then thigh,
    each site accel XYZ then gyro XYZ.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (LENGTH, 12) or not np.isfinite(x).all():
        raise ValueError('Expected finite N x 250 x 12 signals')
    out = np.empty_like(x)
    quality = []
    for site in range(2):
        a, w = x[:, :, site * 6:site * 6 + 3], x[:, :, site * 6 + 3:site * 6 + 6]
        scale = np.median(np.linalg.norm(a, axis=2), axis=1)
        if np.any(scale < 1e-8):
            raise ValueError('Cannot normalize zero acceleration')
        a = a / scale[:, None, None]
        vertical = a.mean(1)
        strength = np.linalg.norm(vertical, axis=1)
        if np.any(strength < 1e-6):
            raise ValueError('Mean acceleration does not establish a stable reference')
        vertical /= strength[:, None]
        transverse = w - np.einsum('nti,ni->nt', w, vertical)[:, :, None] * vertical[:, None, :]
        cov = np.einsum('nti,ntj->nij', transverse, transverse) / LENGTH
        eigenvalues, vectors = np.linalg.eigh(cov)
        lateral = vectors[:, :, -1]
        lateral -= np.sum(lateral * vertical, axis=1)[:, None] * vertical
        lateral_norm = np.linalg.norm(lateral, axis=1)
        if np.any(lateral_norm < 1e-8):
            raise ValueError('No transverse rotation reference')
        lateral /= lateral_norm[:, None]
        projection = np.einsum('nti,ni->nt', transverse, lateral)
        peak = projection[np.arange(len(x)), np.abs(projection).argmax(1)]
        lateral *= np.where(peak < 0, -1., 1.)[:, None]
        forward = np.cross(vertical, lateral)
        basis = np.stack([lateral, forward, vertical], axis=2)
        out[:, :, site * 6:site * 6 + 3] = np.einsum('nti,nij->ntj', a, basis)
        out[:, :, site * 6 + 3:site * 6 + 6] = np.einsum('nti,nij->ntj', w, basis)
        quality.append({
            'median_acceleration_norm': scale,
            'transverse_gyro_rms': np.sqrt(np.maximum(eigenvalues[:, -1], 0)),
            'principal_axis_fraction': eigenvalues[:, -1] / np.maximum(eigenvalues.sum(1), 1e-12),
        })
    return out.astype(np.float32), quality


def standardize(x, mean, scale) -> np.ndarray:
    """Apply fixed checkpoint mean/scale (not a re-fit StandardScaler)."""
    mean = as_numpy(mean).astype(np.float32, copy=False)
    scale = as_numpy(scale).astype(np.float32, copy=False)
    x = np.asarray(x, dtype=np.float32)
    if mean.shape != scale.shape:
        raise ValueError(f'mean/scale shape mismatch: {mean.shape} vs {scale.shape}')
    return (x - mean) / scale


def prepare_canonical100_inputs(x, mean, scale, *, apply_lowpass: bool = True):
    """Filter (optional) → canonicalize → checkpoint standardize.

    Returns standardized float32 windows shaped [N, 250, 12].
    """
    x = np.asarray(x)
    if apply_lowpass:
        x = lowpass_windows(x)
    canonical, _quality = canonicalize(x)
    return standardize(canonical, mean, scale)
