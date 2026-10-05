"""Verify both documented Flask import paths by starting local servers."""
import os
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class StartupTests(unittest.TestCase):
    def check_startup(self, directory, app_name):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                [sys.executable, '-B', '-m', 'flask', '--app', app_name, 'run',
                 '--host', '127.0.0.1', '--port', str(port), '--no-reload'],
                cwd=directory, stdout=output, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', FLASK_DEBUG='0'))
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                            self.assertEqual(response.status, 200)
                            self.assertEqual(json.load(response), {'status': 'ok'})
                            return
                    except OSError:
                        time.sleep(.2)
                output.seek(0)
                self.fail(output.read().decode(errors='replace'))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)

    def test_project_root(self):
        self.check_startup(ROOT, 'backend.app')

    def test_backend_folder(self):
        self.check_startup(ROOT / 'backend', 'app')


if __name__ == '__main__':
    unittest.main()
