import unittest
from tools.export_public_results import redact


class PublicationRedactionTests(unittest.TestCase):
    def test_windows_paths_and_dictionary_keys_are_redacted(self):
        prefixes = [('C:\\Users\\someone\\project', '<project>'), ('C:\\Users\\someone', '<user>')]
        value = {'C:\\Users\\someone\\project\\file.pt': ['C:/Users/someone/other', 42]}
        self.assertEqual(redact(value, prefixes), {'<project>/file.pt': ['<user>/other', 42]})

    def test_longest_prefix_wins_and_original_is_unchanged(self):
        value = {'path': '/user/work/project/a', 'sha': 'deadbeef', 'number': 23.88791025}
        result = redact(value, [('/user','<user>'),('/user/work/project','<project>')])
        self.assertEqual(result['path'], '<project>/a')
        self.assertEqual(value['path'], '/user/work/project/a')
        self.assertEqual(result['sha'], value['sha'])
        self.assertEqual(result['number'], value['number'])

    def test_nonpath_equation_escapes_remain_unchanged(self):
        self.assertEqual(redact('\\alpha = 1', [('/user','<user>')]), '\\alpha = 1')


if __name__=='__main__': unittest.main()
