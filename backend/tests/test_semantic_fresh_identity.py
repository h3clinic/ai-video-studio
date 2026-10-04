import unittest
from cloud.run_semantic_fresh import run_identity, ROOT


class FreshIdentityTests(unittest.TestCase):
    def test_legacy_unchanged(self):
        out,name=run_identity()
        self.assertEqual(out,ROOT/'artifacts/cloud/object_edit_fresh4090_v4')
        self.assertEqual(name,'gaussian-semantic-fresh-20261004-v3')

    def test_safe_new_identity(self):
        out,name=run_identity('apple-agent-v5_20261004')
        self.assertEqual(out,ROOT/'artifacts/cloud/apple-agent-v5_20261004')
        self.assertEqual(name,'apple-agent-v5_20261004')

    def test_paths_and_ambiguous_names_rejected(self):
        for name in ('','../x','a/b','a\\b','A','a b','a'*61,42):
            with self.subTest(name=name),self.assertRaises(ValueError):run_identity(name)


if __name__=='__main__':unittest.main()
