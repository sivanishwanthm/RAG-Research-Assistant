import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app_storage import save_json


class ReportStorageTests(unittest.TestCase):
    def test_save_and_failed_replace_preserve_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            save_json(path, {'status': 'complete'})
            with patch('app_storage.os.replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    save_json(path, {'status': 'partial'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'complete'})
            self.assertEqual(list(Path(directory).iterdir()), [path])
