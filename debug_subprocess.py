import subprocess
import sys
import os

print("Testing subprocess Popen...")
try:
    cmd = "docker --version"
    print(f"Running Popen: {cmd}")
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    print(f"Return code: {p.returncode}")
    print(f"Stdout: {stdout.decode('utf-8', errors='replace')}")
    print(f"Stderr: {stderr.decode('utf-8', errors='replace')}")
except Exception as e:
    print(f"Exception: {e}")
