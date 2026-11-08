import numpy as np
import torch
from torch import nn


def _infer_transformertsai_hparams(state_dict, meta=None):
    """Infer TransformerTSAI constructor args from a checkpoint state_dict."""
    meta = meta or {}
    w_p = state_dict['W_P.weight']  # [d_model, c_in]
    w_pos = state_dict['W_pos']     # [seq_len, d_model]
    d_model, c_in = int(w_p.shape[0]), int(w_p.shape[1])
    seq_len = int(meta.get('window_length', w_pos.shape[0]))
    d_ff = int(state_dict['encoder.layers.0.ff.0.weight'].shape[0])
    layer_ids = {
        int(k.split('.')[2])
        for k in state_dict
        if k.startswith('encoder.layers.') and k.endswith('.self_attn.W_Q.weight')
    }
    n_layers = max(layer_ids) + 1 if layer_ids else 1
    c_out = sum(1 for k in state_dict if k.startswith('fc2_') and k.endswith('.weight'))
    if c_out == 0:
        targets = meta.get('targets')
        c_out = len(targets) if targets else 3
    # Weight shapes do not encode n_heads uniquely; BioMAT / TSAI default is 16.
    n_heads = int(meta.get('n_heads', 16))
    return {
        'c_in': c_in,
        'c_out': c_out,
        'seq_len': seq_len,
        'd_model': d_model,
        'd_ff': d_ff,
        'n_layers': n_layers,
        'n_heads': n_heads,
    }


def _build_transformertsai_from_checkpoint(checkpoint):
    """Rebuild a TransformerTSAI module from a metadata + state_dict checkpoint."""
    from model.transformer_tsai import TransformerTSAI

    state_dict = checkpoint['state_dict']
    meta = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    hp = _infer_transformertsai_hparams(state_dict, meta)
    model = TransformerTSAI(
        c_in=hp['c_in'],
        c_out=hp['c_out'],
        seq_len=hp['seq_len'],
        max_seq_len=hp['seq_len'],
        n_layers=hp['n_layers'],
        d_model=hp['d_model'],
        n_heads=hp['n_heads'],
        d_ff=hp['d_ff'],
        res_dropout=0.1,
        act='gelu',
        fc_dropout=0.0,
        classification=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f'Checkpoint state_dict mismatch (missing={missing}, unexpected={unexpected})'
        )
    model.checkpoint_meta = meta
    return model


def peek_checkpoint_meta(path):
    """Return checkpoint metadata for state_dict dicts or modules with checkpoint_meta."""
    obj = torch.load(
        path,
        map_location=torch.device('cpu'),
        weights_only=False,
    )
    if isinstance(obj, dict) and 'state_dict' in obj:
        return {k: v for k, v in obj.items() if k != 'state_dict'}
    meta = getattr(obj, 'checkpoint_meta', None)
    return dict(meta) if meta else {}


def load_trained_model(path):
    """Load a full pickled model or a state_dict checkpoint with CPU compatibility."""
    obj = torch.load(
        path,
        map_location=torch.device('cpu'),
        weights_only=False,
    )
    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict) and 'state_dict' in obj:
        model = _build_transformertsai_from_checkpoint(obj)
    else:
        raise TypeError(
            f'Unsupported checkpoint type at {path}: {type(obj)!r}. '
            'Expected nn.Module or dict with state_dict.'
        )

    # Older nn.GELU checkpoints lack `approximate` (added in newer PyTorch).
    for module in model.modules():
        if isinstance(module, nn.GELU) and not hasattr(module, 'approximate'):
            module.approximate = 'none'
    return model


# https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
