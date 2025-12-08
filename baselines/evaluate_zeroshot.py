"""
Zero-shot CLIP baseline evaluation.
"""

import os
import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.clip.eval.evaluator import main as evaluate_main

# This script is a wrapper around the main CLIP evaluator
# It's used for zero-shot evaluation (without fine-tuning checkpoint)

if __name__ == "__main__":
    # The main evaluator will be called directly
    # For zero-shot, we just need to not pass a checkpoint
    # or pass "pretrained" as checkpoint
    evaluate_main()
