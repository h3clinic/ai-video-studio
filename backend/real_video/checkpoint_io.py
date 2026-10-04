"""Durable, verified, versioned local checkpoints. Not a hardware-failure guarantee."""
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): value.update(chunk)
    return value.hexdigest()


def _finite(value):
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all(): raise ValueError('Nonfinite checkpoint tensor')
    elif isinstance(value,dict):
        for item in value.values(): _finite(item)
    elif isinstance(value,(tuple,list)):
        for item in value: _finite(item)


def load_verified(path):
    path=Path(path)
    sidecar=path.with_suffix(path.suffix+'.sha256.json')
    if sidecar.exists():
        expected=json.loads(sidecar.read_text(encoding='utf-8'))['sha256']
        if digest(path)!=expected: raise ValueError(f'Checkpoint digest mismatch: {path}')
    result=torch.load(path,map_location='cpu',weights_only=True)
    _finite(result)
    return result


def _atomic(path,writer,validate=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.pending')
    temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:
            writer(stream)
            stream.flush(); os.fsync(stream.fileno())
        if validate: validate(temporary)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def save_training_checkpoint(checkpoint,out,step,is_best):
    out=Path(out)
    version=out/'checkpoints'/f'step_{step:08d}.pt'
    if version.exists(): raise FileExistsError(f'Immutable recovery checkpoint already exists: {version}')
    _atomic(version,lambda stream:torch.save(checkpoint,stream),load_verified)
    checksum=digest(version)
    record=json.dumps(dict(sha256=checksum,step=step,version=str(version.name)),indent=2).encode('utf-8')
    _atomic(version.with_suffix('.pt.sha256.json'),lambda stream:stream.write(record))
    for name in (['last.pt','best.pt'] if is_best else ['last.pt']):
        destination=out/name
        def copy(stream):
            with version.open('rb') as source: shutil.copyfileobj(source,stream,1024*1024)
        _atomic(destination,copy)
        _atomic(destination.with_suffix('.pt.sha256.json'),lambda stream:stream.write(record))
    return checksum


def save_inference_checkpoint(checkpoint,path):
    path=Path(path)
    if path.exists(): raise FileExistsError(f'Refusing to replace exported weights: {path}')
    _atomic(path,lambda stream:torch.save(checkpoint,stream),load_verified)
    checksum=digest(path)
    record=json.dumps(dict(sha256=checksum),indent=2).encode('utf-8')
    _atomic(path.with_suffix(path.suffix+'.sha256.json'),lambda stream:stream.write(record))
    return checksum


@contextmanager
def keep_windows_awake():
    """The training process owns its power request and releases it on exit."""
    function=None
    if os.name=='nt':
        function=ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes=[ctypes.c_uint32]; function.restype=ctypes.c_uint32
        if not function(0x80000003): raise OSError('Windows rejected training sleep prevention')
        print(json.dumps(dict(windows_sleep_prevention=True)),flush=True)
    try: yield
    finally:
        if function:
            function(0x80000000)
            print(json.dumps(dict(windows_sleep_prevention=False)),flush=True)
