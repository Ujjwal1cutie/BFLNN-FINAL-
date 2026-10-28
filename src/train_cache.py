"""Opt-in, content-addressed archive of deterministic Adam training runs.

Set BFLNN_CACHE_DIR to reuse identical runs across ablation scripts. The key
includes the actual basis/derivative matrices, not merely their basis names;
normalisation and monomial monkeypatches therefore cannot collide. Timing in a
cached result is the original training duration, never the cache lookup time.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import scipy


def _array_digest(digest, label, value):
    arr = np.ascontiguousarray(value)
    digest.update(label.encode())
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())


def cached_training(fn):
    signature = inspect.signature(fn)
    source_path = Path(fn.__code__.co_filename).resolve()

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        directory = os.environ.get('BFLNN_CACHE_DIR')
        if not directory:
            return fn(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        arrays = {}
        for grid_name in ('x_train', 'x_val'):
            grid = values.pop(grid_name, None)
            if grid is None:
                continue
            arrays[grid_name] = np.asarray(grid)
            matrices = fn.__globals__['get_basis'](
                values['name'], grid, values['degree'], values['x_max'])
            for kind, matrix in zip(('Phi', 'dPhi', 'd2Phi'), matrices):
                arrays[grid_name + '_' + kind] = np.asarray(matrix)
        metadata = {
            'schema': 1,
            'module': source_path.name,
            'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'parameters': values,
            'timing_definition': 'original uncached Adam training wall seconds',
        }
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        for label, arr in sorted(arrays.items()):
            _array_digest(digest, label, arr)
        key = digest.hexdigest()
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / (key + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                result = (saved['W'].copy(), saved['train_hist'].tolist(),
                          saved['val_hist'].tolist(), saved['hist_epochs'].tolist(),
                          float(saved['elapsed']))
            # Original train resets the legacy RNG and consumes exactly K draws.
            np.random.seed(values['seed'])
            np.random.randn(values['degree'] + 1, 1)
            return result
        result = fn(*args, **kwargs)
        W, th, vh, he, elapsed = result
        metadata.update({
            'cache_key': key, 'elapsed_seconds': float(elapsed),
            'stopped_epoch': int(he[-1]) if len(he) else 0,
            'checkpoint_epoch': int(he[int(np.argmin(vh))]) if len(vh) else
                                (int(he[-1]) if len(he) else 0),
            'validation_loss': float(min(vh)) if len(vh) else None,
        })
        handle, temporary = tempfile.mkstemp(prefix=key + '.', suffix='.npz', dir=root)
        os.close(handle)
        try:
            np.savez_compressed(temporary, W=W, train_hist=np.asarray(th),
                                val_hist=np.asarray(vh), hist_epochs=np.asarray(he),
                                elapsed=np.asarray(elapsed),
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                                **arrays)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return result

    return wrapped
