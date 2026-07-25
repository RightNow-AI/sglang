"""Publication figure generation for ThoughtBench artifacts."""

from .contracts import ProvenanceError, require_provenance
from .pipeline import generate_all

__all__ = ["ProvenanceError", "generate_all", "require_provenance"]
