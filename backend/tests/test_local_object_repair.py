import unittest
from unittest.mock import patch
import numpy as np
from real_video.local_object_repair import repair_selected,fit_bounded_exposure

class LocalRepairTests(unittest.TestCase):
    def test_core_is_excluded_even_when_only_boundary_written(self):
        im=np.zeros((8,8,3),np.uint8);im[2:6,2:6]=[255,80,0]
        exclude=np.zeros((8,8),bool);exclude[2:6,2:6]=True
        write=exclude.copy();write[3:5,3:5]=False
        with patch('real_video.local_object_repair.cv2.inpaint',return_value=np.full_like(im,77)) as f:
            result=repair_selected(im,write,exclude)
        np.testing.assert_array_equal(f.call_args.args[1],exclude.astype('uint8')*255)
        np.testing.assert_array_equal(result[~write],im[~write])
        self.assertTrue((result[write]==77).all())
    def test_write_guard(self):
        with self.assertRaises(ValueError):repair_selected(np.zeros((2,2,3),np.uint8),np.ones((2,2),bool),np.zeros((2,2),bool))
    def test_exposure_bounded_and_deterministic(self):
        c=np.array([[.2,.1,.1],[.3,.1,.1]],np.float32);r=np.ones((4,3),np.float32)*.8
        original=c.copy()
        output,gain=fit_bounded_exposure(c,r)
        self.assertEqual(gain,2.5);self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(c,original)
        np.testing.assert_array_equal(output,fit_bounded_exposure(c,r)[0])
    def test_black_stays_finite(self):
        output,gain=fit_bounded_exposure(np.zeros((2,3)),np.ones((2,3)))
        self.assertEqual(gain,1);self.assertTrue((output==0).all())

if __name__=='__main__':unittest.main()
