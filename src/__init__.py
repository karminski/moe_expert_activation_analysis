"""
MoE Expert Activation Analysis Tool

This package provides tools to analyze expert activation patterns in Mixture-of-Experts (MoE) models.
Supports analysis of activation probabilities, layer correlations, and periodic patterns.

Supported models:
- MiniMax-M2
- DeepSeek-V3
- Qwen3 MoE variants
"""

__version__ = "0.1.0"

try:
    from .moe_analyzer import MoEAnalyzer
    from .moe_hooks import MoEHookManager
    from .visualizer import MoEVisualizer
except ImportError:
    from moe_analyzer import MoEAnalyzer
    from moe_hooks import MoEHookManager
    from visualizer import MoEVisualizer

__all__ = ["MoEAnalyzer", "MoEHookManager", "MoEVisualizer"]

