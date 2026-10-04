import unittest
from PIL import Image
from real_video.zero123_views import prepare_input,split_views

class ZeroViewTests(unittest.TestCase):
    def test_aspect_preserved_and_alpha_retained(self):
        image=Image.new('RGBA',(80,40),(255,0,0,255))
        out=prepare_input(image,[0,0,80,40],96)
        self.assertEqual(out.size,(96,96));self.assertEqual(out.getpixel((0,0))[3],0)
        self.assertEqual(out.getpixel((48,48)),(255,0,0,255))
    def test_invalid_roi_rejected(self):
        with self.assertRaises(ValueError):prepare_input(Image.new('RGBA',(10,10)),[0,0,11,10])
    def test_tile_order(self):
        im=Image.new('RGB',(640,960))
        for i in range(6):im.paste((i*40,0,0),((i%2)*320,(i//2)*320,(i%2+1)*320,(i//2+1)*320))
        self.assertEqual([v.getpixel((0,0))[0] for v in split_views(im)],[i*40 for i in range(6)])
    def test_wrong_output_shape_rejected(self):
        with self.assertRaises(ValueError):split_views(Image.new('RGB',(960,640)))
