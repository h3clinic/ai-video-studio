import unittest
from PIL import Image
from real_video.region_detail import crop_region

class RegionDetailTests(unittest.TestCase):
    def test_crop_keeps_alpha_and_coordinates(self):
        image=Image.new('RGBA',(100,80),(1,2,3,42))
        crop=crop_region(image,[10,20,70,60])
        self.assertEqual(crop.size,(60,40));self.assertEqual(crop.getpixel((0,0)),(1,2,3,42))
    def test_invalid_roi(self):
        image=Image.new('RGBA',(100,80))
        for box in [[0,0,0,4],[-1,0,4,4],[0,0,101,2],[0,0,4,81]]:
            with self.assertRaises(ValueError):crop_region(image,box)
