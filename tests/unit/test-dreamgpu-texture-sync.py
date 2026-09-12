#!/usr/bin/env python3
"""Exercise the actual Rust dependency engine after removal from platform C."""
from pathlib import Path
import os
import subprocess

workspace = Path(__file__).resolve().parents[4]
subprocess.run([
    os.environ.get('CARGO', 'cargo'), 'test', '--offline', '--locked',
    '--manifest-path', str(workspace / 'Cargo.toml'), '-p', 'dreamgpu-host',
    'texture::tests::same_context_and_foreign_version_fence_ownership',
    '--', '--exact',
], check=True)
