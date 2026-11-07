import json
import yaml

import torch


def _resolve_device(device):
    """Use CUDA only when the installed torch build supports it."""
    if isinstance(device, str) and device.startswith('cuda') and not torch.cuda.is_available():
        return 'cpu'
    return device


def get_config_universal(dataset_name):
    with open(f'./configs/{dataset_name}_config.json') as f:
        config = json.load(f)
    config['device'] = _resolve_device(config.get('device', 'cpu'))
    return config


def get_sweep_config_universal(dataset_name):
    with open(f'./configs/sweep_{dataset_name}_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_model_config(model_config):
    with open(f'./configs/{model_config}.json') as f:
        config = json.load(f)
    return config
