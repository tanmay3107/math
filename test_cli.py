import unittest
from unittest.mock import patch
import io
import os
import tempfile
from cli import main

class TestCLI(unittest.TestCase):
    """Automated unit tests for the VectorStore CLI interface using mocks."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_add_and_get(self, mock_stdout, mock_input):
        # Simulate sequential user command inputs ending with 'exit'
        mock_input.side_effect = [
            'add doc1 1.0,0.0,0.0 {"topic":"tech"}',
            'get doc1',
            'exit'
        ]
        main()
        output = mock_stdout.getvalue()

        self.assertIn("Added vector 'doc1'.", output)
        self.assertIn('"id": "doc1"', output)
        self.assertIn('"topic": "tech"', output)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_search(self, mock_stdout, mock_input):
        mock_input.side_effect = [
            'add doc1 1.0,0.0,0.0',
            'add doc2 0.0,1.0,0.0',
            'search 1.0,0.0,0.0 1 cosine',
            'exit'
        ]
        main()
        output = mock_stdout.getvalue()

        self.assertIn("Top 1 results using 'cosine':", output)
        self.assertIn("[doc1]", output)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_delete(self, mock_stdout, mock_input):
        mock_input.side_effect = [
            'add doc1 1.0,0.0,0.0',
            'delete doc1',
            'get doc1',
            'exit'
        ]
        main()
        output = mock_stdout.getvalue()

        self.assertIn("Deleted vector 'doc1'.", output)
        self.assertIn("Vector ID 'doc1' not found.", output)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_save_and_load(self, mock_stdout, mock_input):
        filepath = os.path.join(self.temp_dir.name, "cli_store.json")
        mock_input.side_effect = [
            'add doc1 1.0,0.0,0.0',
            f'save {filepath}',
            'exit'
        ]
        main()

        # Reload state in a fresh CLI session
        with patch("builtins.input", side_effect=[f'load {filepath}', 'get doc1', 'exit']), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout_load:
            main()
            load_output = mock_stdout_load.getvalue()
            self.assertIn(f"Loaded store from '{filepath}'.", load_output)
            self.assertIn('"id": "doc1"', load_output)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_invalid_command_and_help(self, mock_stdout, mock_input):
        mock_input.side_effect = [
            'help',
            'invalidcmd',
            'exit'
        ]
        main()
        output = mock_stdout.getvalue()

        self.assertIn("VectorStore CLI Commands", output)
        self.assertIn("Unknown command: 'invalidcmd'.", output)

if __name__ == "__main__":
    unittest.main()