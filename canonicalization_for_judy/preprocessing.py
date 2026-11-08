"""100 Hz preprocessing: same canonical representation as model training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import torch

HZ, LENGTH = 100, 250
SITES = ['shank', 'thigh']
TARGETS = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
CLASSES = ['walking', 'ramp_ascent', 'ramp_descent', 'stair_ascent', 'stair_descent']
PREPROCESS = 'canonical100_v1: 20Hz lowpass; 250 samples; median acceleration norm; mean-acceleration + transverse-gyro PCA basis; peak-projection sign'


def canonicalize(x):
    """Same deterministic, label-free rotation/acceleration-scale normalization per window.

This frame is signal-derived, NOT an anatomical calibration. Its sign may change
between windows. All predictions and validation use this same representation.
"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (LENGTH, 12) or not np.isfinite(x).all():
        raise ValueError('Expected finite N x 250 x 12 signals')
    out = np.empty_like(x)
    quality = []
    for site in range(2):
        a, w = x[:, :, site*6:site*6+3], x[:, :, site*6+3:site*6+6]
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
        out[:, :, site*6:site*6+3] = np.einsum('nti,nij->ntj', a, basis)
        out[:, :, site*6+3:site*6+6] = np.einsum('nti,nij->ntj', w, basis)
        quality.append({'median_acceleration_norm': scale,
                        'transverse_gyro_rms': np.sqrt(np.maximum(eigenvalues[:, -1], 0)),
                        'principal_axis_fraction': eigenvalues[:, -1] / np.maximum(eigenvalues.sum(1), 1e-12)})
    return out.astype(np.float32), quality

def read_phone(path):
    df = pd.read_csv(path)
    columns = ['common_timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    values = df[columns].to_numpy(dtype=np.float64)
    if len(values) < LENGTH or not np.isfinite(values).all():
        raise ValueError('Missing/invalid phone samples')
    if set(df.accel_units) not in ({'m/s^2'}, {'g'}) or set(df.gyro_units) not in ({'deg/s'}, {'rad/s'}):
        raise ValueError('Unexpected phone units')
    t, x = values[:, 0], values[:, 1:]
    dt = np.diff(t)
    rate = (len(t)-1)/(t[-1]-t[0])
    if (dt <= 0).any() or dt.max() > .05 or not 90 <= rate <= 110:
        raise ValueError('This adapter requires a continuous approximately 100 Hz recording')
    if set(df.accel_units) == {'g'}:
        x[:, :3] *= 9.80665
    if set(df.gyro_units) == {'deg/s'}:
        x[:, 3:] = np.deg2rad(x[:, 3:])
    return t, x, rate, df

def prepare_phone(session):
    # Phone 1 is thigh; phone 2 is shank. Model ordering is shank then thigh.
    loaded = [read_phone(session / name) for name in ['phone_2.csv', 'phone_1.csv']]
    metadata_path = session / 'metadata.json'
    origin = float(json.loads(metadata_path.read_text())['start']) if metadata_path.exists() else max(v[0][0] for v in loaded)
    start, stop = max(a[0][0] for a in loaded), min(a[0][-1] for a in loaded)
    timeline = start + np.arange(int(np.floor((stop-start)*HZ))+1)/HZ
    if len(timeline) < LENGTH:
        raise ValueError('Insufficient synchronized overlap')
    x = np.concatenate([np.column_stack([np.interp(timeline,t,v[:,j]) for j in range(6)])
                        for t,v,_,_ in loaded],axis=1)
    x = sosfiltfilt(butter(4,20,fs=HZ,output='sos'),x,axis=0)
    starts = np.arange(0, len(x)-LENGTH+1, 50)
    windows = np.stack([x[s:s+LENGTH] for s in starts])
    canonical, quality = canonicalize(windows)
    # Fixed acquisition/low-motion gate, not a learned standing class or OOD detector.
    moving = np.ones(len(windows), dtype=bool)
    for site in range(2):
        gyro = windows[:,:,site*6+3:site*6+6]
        fraction = (np.linalg.norm(gyro,axis=2) > .25).mean(1)
        moving &= fraction >= .75
        moving &= quality[site]['principal_axis_fraction'] >= .6
    info = {'rates_hz': {site: v[2] for site,v in zip(SITES,loaded)},
            'input_sha256': {name:hashlib.sha256((session/name).read_bytes()).hexdigest()
                             for name in ['phone_1.csv','phone_2.csv']},
            'session_origin_host_s':origin, 'gating': 'Both phones >=75% samples angular speed >0.25 rad/s; transverse gyro principal-axis fraction >=0.6',
            'quality': [{key:val.tolist() for key,val in q.items()} for q in quality]}
    return canonical, starts, timeline-origin, moving, info

@torch.inference_mode()
def predict(model, x, mean, scale, device, task):
    model.eval()
    out = []
    for start in range(0, len(x), 64):
        batch = torch.from_numpy((x[start:start+64] - mean) / scale).to(device)
        value = model(batch)
        if task == 'classifier':
            value = value.softmax(-1)
        out.append(value.cpu().numpy())
    return np.concatenate(out)
