import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import find_categories as fc


@mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""})
class LoadApiKeyTest(unittest.TestCase):
    def env_file(self, text, encoding):
        path = Path(tempfile.mkdtemp()) / ".env"
        path.write_text(text, encoding=encoding)
        return path

    def test_plain_utf8(self):
        self.assertEqual(fc.load_api_key(self.env_file("OPENAI_API_KEY=sk-test-1\n", "utf-8")), "sk-test-1")

    def test_utf8_with_bom_as_saved_by_notepad(self):
        self.assertEqual(fc.load_api_key(self.env_file("OPENAI_API_KEY=sk-test-1\n", "utf-8-sig")), "sk-test-1")

    def test_missing_key_is_reported(self):
        with self.assertRaisesRegex(fc.SetupError, "OPENAI_API_KEY is not set"):
            fc.load_api_key(self.env_file("OPENAI_API_KEY=\n", "utf-8"))


if __name__ == "__main__":
    unittest.main()
