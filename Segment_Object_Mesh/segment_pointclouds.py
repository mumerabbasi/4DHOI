"""Backward-compatible wrapper for the renamed mesh segmentation script.

Use `segment_meshes.py` for all new runs.
"""

from segment_meshes import main


if __name__ == "__main__":
    main()
