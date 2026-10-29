"""Kept for compatibility: python main.py <input_dir> <output_dir> == python inference/infer.py <input_dir> <output_dir>."""
import os
import runpy
import sys

sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference", "infer.py")
runpy.run_path(sys.argv[0], run_name="__main__")
