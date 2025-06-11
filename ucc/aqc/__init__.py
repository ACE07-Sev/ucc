import importlib

__all__ = ["calculate_entanglement_entropy_slope", "MPS_Encoder"]

from .utils import calculate_entanglement_entropy_slope

# Check if `qmprs` is installed for AQC compilation
qmprs_available = importlib.util.find_spec("qmprs") is not None
if qmprs_available:
    from .qmprs_compiler import QmprsCompiler as MPS_Encoder
else:
    from .mps_sequential import Sequential as MPS_Encoder
