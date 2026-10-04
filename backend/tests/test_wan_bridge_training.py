import unittest
from argparse import Namespace
from real_video.train_wan_bridge import selected_entries, cache_path


class BridgeSplitTests(unittest.TestCase):
    def test_expanded_high_resolution_cache_cannot_reuse_pilot(self):
        pilot = cache_path(Namespace(all_clips=False, size=128))
        expanded = cache_path(Namespace(all_clips=True, size=256))
        self.assertNotEqual(pilot, expanded)
        self.assertEqual(pilot.name, 'wan_bridge_v1.pt')
        self.assertEqual(expanded.name, 'wan_bridge_all_256.pt')

    def test_selection_never_uses_test_and_is_input_order_independent(self):
        entries = [dict(file=f'{split}_{label}_{i}.avi', label=label, split=split, group=f'{split}_{label}_{i}')
                   for split in ['train', 'validation', 'test'] for label in ['a', 'b'] for i in range(10)]
        a = selected_entries(dict(labels=['a', 'b'], entries=entries))
        b = selected_entries(dict(labels=['a', 'b'], entries=entries[::-1]))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)
        self.assertEqual(sum(e['split']=='train' for e in a), 16)
        self.assertFalse(any(e['split']=='test' for e in a))
        expanded = selected_entries(dict(labels=['a', 'b'], entries=entries), expanded=True)
        self.assertEqual(len(expanded), 40)
        self.assertFalse(any(e['split']=='test' for e in expanded))


if __name__ == '__main__':
    unittest.main()
