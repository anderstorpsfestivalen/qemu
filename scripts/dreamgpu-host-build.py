#!/usr/bin/env python3
"""Build the allocation-free host archive in a separate Cargo target directory."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--cargo', required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--target-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
env = os.environ.copy()
env['CARGO_PROFILE_RELEASE_PANIC'] = 'abort'
# An enclosing Cargo build may set target/profile flags for a different crate.
# Native compilation here must use this QEMU host, never a guest CPU target.
env.pop('CARGO_BUILD_TARGET', None)
subprocess.run([a.cargo, 'build', '--offline', '--locked', '--release',
                '--manifest-path', str(a.manifest), '--target-dir', str(a.target_dir),
                '-p', 'dreamgpu-host'], check=True, env=env)
shutil.copyfile(a.target_dir / 'release/libdreamgpu_host.a', a.output)
