import tempfile
import unittest
from pathlib import Path
import torch
from gv.codec import encode,decode
from gv.data import make_batch


class CodecTests(unittest.TestCase):
    def test_roundtrip_and_size(self):
        _,_,seq=make_batch(1,batch=1,n=10,steps=24)
        states=[{k:v[0] for k,v in s.items()} for s in seq]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stream.npz'
            encode(states,path)
            restored=list(decode(path))
            self.assertEqual(len(restored),len(states))
            self.assertLess((restored[-1]['p']-states[-1]['p']).abs().max().item(),2e-4)
            self.assertLess((restored[-1]['R']-states[-1]['R']).abs().max().item(),3e-4)

    def test_reject_overflow(self):
        _,_,seq=make_batch(1,batch=1,n=3,steps=1)
        states=[{k:v[0] for k,v in s.items()} for s in seq]
        states[1]['p']=states[1]['p']+100
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): encode(states,Path(directory)/'bad.npz')
