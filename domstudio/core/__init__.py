"""Headless scientific core for fracture evidence and vector tracing.

Only NumPy and SciPy are required for the default pipeline. Verified learned
model runtimes live behind the optional ``domstudio.learning`` package.
"""

from .detectors import (
    DetectorConfig,
    ModelDetector,
    ModelDetectorConfig,
    MultiscaleFractureDetector,
    PredictorOutput,
)
from .models import EvidenceDetector, EvidenceMap, PathLike, PixelXY, RasterDocument
from .tracing import (
    CancelCallback,
    CorridorAStarTracer,
    ProgressCallback,
    TraceConfig,
    TraceProgress,
    TraceRequest,
    TraceResult,
    TraceStatus,
    simplify_polyline,
)
from .vectorization import (
    FractureVectorizer,
    VectorizationResult,
    VectorizationSettings,
)

__all__ = [
    "CancelCallback",
    "CorridorAStarTracer",
    "DetectorConfig",
    "EvidenceDetector",
    "EvidenceMap",
    "FractureVectorizer",
    "ModelDetector",
    "ModelDetectorConfig",
    "MultiscaleFractureDetector",
    "PathLike",
    "PixelXY",
    "PredictorOutput",
    "ProgressCallback",
    "RasterDocument",
    "TraceConfig",
    "TraceProgress",
    "TraceRequest",
    "TraceResult",
    "TraceStatus",
    "VectorizationResult",
    "VectorizationSettings",
    "simplify_polyline",
]
