#!/usr/bin/env python3
"""
Simple script to run all training methods in sequence.
"""

import subprocess
import sys


def run_command(cmd):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)
    print("✓ Completed successfully\n")


def main():
    print("Starting sequential training of all models...\n")

    run_command(["python", "train_tfidf.py"])
    run_command(["python", "train_model2vec.py", "minishlab/potion-base-32M"])
    run_command(["python", "train_setfit.py", "BAAI/bge-small-en-v1.5"])

    print("All training completed successfully! 🎉")


if __name__ == "__main__":
    main()
