import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from real_video.representation import refresh_coverage_positions, encode_clip
from real_video.gaussian_edit_workers import validate_packet, iter_rendered_frames
from tests.test_gaussian_edit_workers import fixture


class CoverageRefreshTests(unittest.TestCase):
    def test_uniform_sampling_is_noop(self):
        yy,xx=np.meshgrid(np.arange(8,dtype=np.float32)*4+1.5,np.arange(8,dtype=np.float32)*4+1.5,indexing='ij')
        x,y,reset=refresh_coverage_positions(xx,yy,32,8)
        np.testing.assert_array_equal(x,xx)
        np.testing.assert_array_equal(y,yy)
        self.assertFalse(reset.any())

    def test_collisions_reseed_only_redundant_slots_to_all_cells(self):
        px=np.full((8,8),1.5,np.float32)
        py=px.copy()
        x,y,reset=refresh_coverage_positions(px,py,32,8)
        cells=np.floor((y+.5)/4).astype(int)*8+np.floor((x+.5)/4).astype(int)
        self.assertEqual(len(np.unique(cells)),64)
        self.assertEqual(int(reset.sum()),63)
        self.assertEqual(x[0,0],1.5)
        np.testing.assert_array_equal(px,np.full((8,8),1.5,np.float32))

    def test_nonfinite_positions_rejected(self):
        with self.assertRaises(ValueError):
            refresh_coverage_positions(np.full((8,8),np.nan),np.zeros((8,8)),32,8)

    def test_track_generation_records_reseeding_and_default_unchanged(self):
        images=np.zeros((3,32,32,3),np.uint8)
        flow=np.zeros((32,32,2),np.float32)
        flow[:,:,0]=4
        with patch('cv2.calcOpticalFlowFarneback',return_value=flow):
            legacy=encode_clip(images,grid=8)
            legacy2,g0=encode_clip(images,grid=8,return_generations=True)
            result,g=encode_clip(images,grid=8,refresh_coverage=True,return_generations=True)
        np.testing.assert_array_equal(legacy,legacy2)
        self.assertFalse(g0.any())
        self.assertFalse(g[0].any())
        self.assertGreater(int(g[-1].sum()),0)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((g[1:]-g[:-1])<=1).all())

    def test_refresh_packet_requires_track_versions(self):
        packet=fixture()
        packet['coverage_refresh']=True
        with self.assertRaises(ValueError): validate_packet(packet)
        packet['track_generations']=torch.zeros(4,8,8,dtype=torch.int32)
        validate_packet(packet)
        for invalid in (-1,2):
            bad=copy.deepcopy(packet)
            bad['track_generations'][2,0,0]=invalid
            with self.assertRaises(ValueError): validate_packet(bad)
        with patch('cv2.VideoCapture',side_effect=AssertionError('source access')):
            self.assertEqual(len(list(iter_rendered_frames(packet))),4)


if __name__=='__main__':unittest.main()
