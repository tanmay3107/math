import unittest
from unittest.mock import patch
import io
import os
import tempfile
from cli import main

class TestCLI(unittest.TestCase):
    """Automated unit tests for VectorStore CLI including HNSW, Keyword, Hybrid, and Benchmark commands."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_add_and_get(self, mock_stdout, mock_input):
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
    def test_cli_search_backends(self, mock_stdout, mock_input):
        mock_input.side_effect = [
            'add doc1 1.0,0.0,0.0',
            'search 1.0,0.0,0.0 1 cosine exact',
            'search 1.0,0.0,0.0 1 cosine hnsw',
            'exit'
        ]
        main()
        output = mock_stdout.getvalue()

        self.assertIn("Top 1 results (cosine, exact backend):", output)
        self.assertIn("Top 1 results (cosine, hnsw backend):", output)
        self.assertIn("[doc1]", output)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_keyword_search(self, mock_stdout, mock_input):
        mock_input.side_effect = [
            'add doc1 1.0,0.0,0.0 {"content":"machine learning python"}',
            'keyword machine 1',
            'exit'
        ]
        main()
        output = mock_stdout.getvalue()

        self.assertIn("Top 1 BM25 results for 'machine':", output)
        self.assertIn("[doc1]", output)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_hybrid_search(self, mock_stdout, mock_input):
        mock_input.side_effect = [
            'add doc1 1.0,0.0,0.0 {"content":"machine learning"}',
            'hybrid 1.0,0.0,0.0 machine 1 exact',
            'exit'
        ]
        main()
        output = mock_stdout.getvalue()

        self.assertIn("Top 1 hybrid results (exact backend):", output)
        self.assertIn("[doc1]", output)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_toggle_hnsw(self, mock_stdout, mock_input):
        mock_input.side_effect = [
            'toggle-hnsw',
            'toggle-hnsw',
            'exit'
        ]
        main()
        output = mock_stdout.getvalue()

        self.assertIn("HNSW search backend is now DISABLED.", output)
        self.assertIn("HNSW search backend is now ENABLED.", output)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cli_benchmark(self, mock_stdout, mock_input):
        mock_input.side_effect = [
            'benchmark 20 8',
            'exit'
        ]
        main()
        output = mock_stdout.getvalue()

        self.assertIn("--- Benchmark Summary ---", output)
        self.assertIn("Dataset Size:        20 vectors", output)

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

        with patch("builtins.input", side_effect=[f'load {filepath}', 'get doc1', 'exit']), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout_load:
            main()
            load_output = mock_stdout_load.getvalue()
            self.assertIn(f"Loaded store state from '{filepath}'.", load_output)
            self.assertIn('"id": "doc1"', load_output)

if __name__ == "__main__":
    unittest.main()