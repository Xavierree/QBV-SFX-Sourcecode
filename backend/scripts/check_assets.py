#!/usr/bin/env python3
"""Lists the model/data files the backend needs and which of them are missing (exit code 1 if any). Stdlib only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sfxdeploy import assetcheck
assetcheck.check_or_exit()
print(f'all {len(assetcheck.REQUIRED)} required assets present')
