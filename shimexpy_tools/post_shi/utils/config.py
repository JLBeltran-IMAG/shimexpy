"""Configuration module for SHI processing."""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SHIConfig:
    """Configuration class for SHI processing."""
    @property
    def CONTRAST_TYPES(self) -> List[str]:
        """Get the list of contrast types."""
        return ["absorption", "scattering", "phase", "phasemap"]

    # Directories
    @property
    def BASE_DIR(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def SCRIPT_DIR(self) -> Path:
        return self.BASE_DIR / "scripts"

    @property
    def CACHE_DIR(self) -> Path:
        return self.BASE_DIR / "cache"

    @property
    def TMP_DIR(self) -> Path:
        return self.BASE_DIR / "tmp"

    @property
    def SRC_DIR(self) -> Path:
        return self.BASE_DIR / "src"

    # Phase unwrapping methods
    @property
    def UNWRAP_METHODS(self) -> Dict[str, str]:
        return {
            "branch_cut": "Goldstein's Branch-Cut Unwrapping (Simplified)",
            "least_squares": "Least-Squares Phase Unwrapping using FFT",
            "quality_guided": "Quality-Guided Phase Unwrapping",
            "min_lp": "Minimum Lp-Norm Phase Unwrapping",
            "": "Algorithm based on sorting by reliability following a noncontinuous path"
        }

    def validate_unwrap_method(self, method: str) -> bool:
        """Validate if the given unwrap method is supported."""
        return method in self.UNWRAP_METHODS or not method

    def get_unwrap_description(self, method: str) -> str:
        """Get the description of the unwrap method."""
        return self.UNWRAP_METHODS.get(method, self.UNWRAP_METHODS[""])

# Global instance
config = SHIConfig()
