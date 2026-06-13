"""Pytest path setup.

The test suite mixes two import styles:

* ``test_statistical_validation.py`` imports via the package, e.g.
  ``from evaluation.statistical_framework_v2 import ...`` (needs the repo
  root on ``sys.path``).
* ``test_suite.py`` imports bare modules, e.g. ``from utils import ...`` and
  ``from models.model import Model`` (needs the ``evaluation/`` directory on
  ``sys.path``).

Adding both here lets the full suite collect and run without per-file path
hacks or changing any test logic.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_EVALUATION = os.path.join(_ROOT, "evaluation")

for _p in (_ROOT, _EVALUATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)
