import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import external_validation as ev


class ExternalValidationTests(unittest.TestCase):
    def test_selection_reproducible_independent_of_input_order(self):
        rows=[dict(uuid=str(i),prompt='an external prompt',**{k:'0.001' for k in
              ('toxicity','obscene','identity_attack','insult','threat','sexual_explicit')}) for i in range(100)]
        rows[0]['sexual_explicit']='0.9'
        eligible,a=ev.select_cases(rows)
        _,b=ev.select_cases(list(reversed(rows)))
        self.assertEqual([x['uuid'] for x in a],[x['uuid'] for x in b])
        self.assertEqual(len(eligible),99)
        self.assertEqual(len(a),8)

    def test_tar_paths_never_control_extraction_destination(self):
        uid='776d0a24-9c93-5336-9ff4-d975aad0f98d'
        missing='00000000-0000-0000-0000-000000000000'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.tar'
            with tarfile.open(source,'w') as archive:
                info=tarfile.TarInfo(f'../../outside/{uid}.mp4'); info.size=4
                archive.addfile(info,io.BytesIO(b'data'))
            with patch.object(ev,'OUT',root):
                records=ev.extract_selected(source,'baseline',dict(selected=[dict(uuid=uid),dict(uuid=missing)]))
            self.assertEqual((root/'baseline'/f'{uid}.mp4').read_bytes(),b'data')
            self.assertEqual(sum(x.get('status')=='missing' for x in records),1)
            self.assertTrue((root/'baseline_extraction.json').exists())


if __name__=='__main__': unittest.main()
