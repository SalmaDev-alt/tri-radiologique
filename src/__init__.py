"""Package source du projet de tri radiologique."""
# IMPORTANT (Windows) : on importe torch AVANT numpy/scipy/matplotlib pour éviter
# un conflit de bibliothèques natives OpenMP (WinError 1114 sur c10.dll).
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401  (chargé tôt volontairement, ne pas retirer)
except Exception:
    pass
