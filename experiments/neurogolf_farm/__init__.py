"""NeuroGolf compiler farm support package."""

from .cost_extractor import CostExtractor, CostSignal
from .bundle_manager import BundleCandidate, BundleLedger
from .ir import IRNode, IRProgram, PrimitiveKind, TensorSpec
from .public_code import PublicCodeRegistry, PublicCodeSource

# The IR, cost guardrail, bundle ledger, and public-code registry are pure
# Python and must stay importable even when onnx/onnxruntime are absent (the
# cost extractor is explicitly a "cheap pre-emission screen"). Only the ONNX
# emitter and the farm runner need the onnx toolchain, so import them lazily and
# expose `ONNX_TOOLCHAIN_AVAILABLE` so callers can degrade instead of crashing.
try:
    from .emitter import emit_candidate
    from .farm_runner import FarmConfig, FarmRunner

    ONNX_TOOLCHAIN_AVAILABLE = True
except ModuleNotFoundError:  # onnx not installed in this environment
    emit_candidate = None  # type: ignore[assignment]
    FarmConfig = None  # type: ignore[assignment]
    FarmRunner = None  # type: ignore[assignment]
    ONNX_TOOLCHAIN_AVAILABLE = False

__all__ = [
    "CostExtractor",
    "CostSignal",
    "BundleCandidate",
    "BundleLedger",
    "emit_candidate",
    "FarmConfig",
    "FarmRunner",
    "ONNX_TOOLCHAIN_AVAILABLE",
    "IRNode",
    "IRProgram",
    "PrimitiveKind",
    "PublicCodeRegistry",
    "PublicCodeSource",
    "TensorSpec",
]
