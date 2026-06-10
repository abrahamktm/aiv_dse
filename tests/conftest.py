import os
from pathlib import Path

# Ensure tests run with the project root as CWD so relative paths
# like "samples/report_pass.json" and "policy/default_policy.yaml" work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
