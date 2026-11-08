"""Run the trained 100 Hz shank/thigh models, or verify their saved test outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from classifier import MovementCNN
from model import BioMAT
from preprocessing import HZ, LENGTH, SITES, TARGETS, CLASSES, PREPROCESS, prepare_phone, predict

BASE = Path(__file__).resolve().parent


def load_models():
    models = {}
    for task, filename in [('regressor', 'model.pt'), ('classifier', 'classifier_model.pt')]:
        saved = torch.load(BASE / filename, weights_only=True, map_location='cpu')
        expected = {'sample_rate': HZ, 'window_length': LENGTH, 'sites': SITES,
                    'preprocess': PREPROCESS, 'task': task, 'targets': TARGETS, 'classes': CLASSES}
        for key, value in expected.items():
            if saved[key] != value:
                raise ValueError(f'{filename}: unexpected {key}; use the 100 Hz checkpoint supplied here')
        model = BioMAT(seq_len=LENGTH, features=12) if task == 'regressor' else MovementCNN(12, 5)
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval()
        models[task] = (model, saved['mean'].numpy(), saved['scale'].numpy())
    return models


def infer(models, x):
    if x.ndim != 3 or x.shape[1:] != (250, 12) or not np.isfinite(x).all():
        raise ValueError('Expected finite canonicalized windows shaped [N, 250, 12]')
    outputs = {task: predict(model, x, mean, scale, 'cpu', task)
               for task, (model, mean, scale) in models.items()}
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise ValueError('Nonfinite model output')
    return outputs


def verify(models):
    with np.load(BASE / 'verification_examples.npz', allow_pickle=False) as example:
        output = infer(models, example['x'])
        np.testing.assert_allclose(output['regressor'], example['angles'], rtol=0, atol=0.001)
        np.testing.assert_allclose(output['classifier'], example['probabilities'], rtol=0, atol=1e-5)
        np.testing.assert_array_equal(output['classifier'].argmax(1), example['probabilities'].argmax(1))
        return {'verification': 'passed', 'examples': len(example['x']),
                'maximum_angle_difference_degrees': float(np.max(np.abs(output['regressor'] - example['angles']))),
                'maximum_classifier_score_difference': float(np.max(np.abs(output['classifier'] - example['probabilities']))),
                'reference': 'Original saved Camargo test predictions, not phone data',
                'angle_tolerance_degrees': 0.001, 'classifier_score_tolerance': 1e-5}


def run_session(models, session, destination):
    x, starts, timeline, usable, info = prepare_phone(session)
    output = infer(models, x)
    destination.mkdir(parents=True, exist_ok=True)
    for task, values in output.items():
        np.savez(destination / (task + '_window_predictions.npz'), predictions=values,
                 usable=usable, start_session_s=timeline[starts])
    probabilities = output['classifier']
    labels = probabilities.argmax(1)
    activities = pd.DataFrame({'start_session_s': timeline[starts],
                               'end_session_s': timeline[starts] + (LENGTH - 1) / HZ,
                               'usable_motion_window': usable,
                               'candidate_activity': [CLASSES[i] if ok else 'not_evaluated_low_motion_or_unstable_axis'
                                                      for i, ok in zip(labels, usable)],
                               'uncalibrated_model_score': np.where(usable, probabilities.max(1), np.nan)})
    for i, name in enumerate(CLASSES):
        activities[name + '_score'] = np.where(usable, probabilities[:, i], np.nan)
    activities.to_csv(destination / 'activity_windows.csv', index=False)
    sums = np.zeros((len(timeline), 3))
    counts = np.zeros(len(timeline), dtype=int)
    for k, start in enumerate(starts):
        if usable[k]:
            sums[start:start + LENGTH] += output['regressor'][k]
            counts[start:start + LENGTH] += 1
    average = np.full_like(sums, np.nan)
    np.divide(sums, counts[:, None], out=average, where=counts[:, None] > 0)
    angles = pd.DataFrame({'session_time_s': timeline, 'contributing_windows': counts})
    for j, name in enumerate(TARGETS):
        angles[name + '_degrees_unvalidated'] = average[:, j]
    angles.to_csv(destination / 'joint_angles_experimental.csv', index=False)
    report = {'windows': len(x), 'accepted_windows': int(usable.sum()),
              'candidate_activity_counts': {name: int(np.sum(usable & (labels == i))) for i, name in enumerate(CLASSES)},
              'phone_accuracy_validated': False, 'joint_angle_error_measurable': False,
              'acquisition': info}
    (destination / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'acquisition'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Check bundled Camargo reference outputs')
    parser.add_argument('--session', type=Path, help='Folder with phone_1.csv (thigh), phone_2.csv (shank)')
    parser.add_argument('--output', type=Path, default=BASE / 'output')
    args = parser.parse_args()
    if not args.verify and args.session is None:
        parser.error('Use --verify or --session PATH')
    torch.set_num_threads(4)
    models = load_models()
    if args.verify:
        print(json.dumps(verify(models), indent=2))
    if args.session is not None:
        run_session(models, args.session.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
