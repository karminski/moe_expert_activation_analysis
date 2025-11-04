# Changelog

All notable changes to the MoE Expert Activation Analysis tool will be documented in this file.

## [Unreleased]

### Added - 2025-10-31

#### 3D Expert Activation Visualization
- **Interactive 3D surface plots** of expert activations
- Visualizes three dimensions simultaneously:
  - Expert index (X-axis)
  - Token position (Y-axis)
  - Activation probability (Z-axis/height)
- **Multiple layers stacked** with offsets for comparison
- **Interactive features:**
  - Rotate, zoom, and pan the 3D view
  - Hover for detailed values
  - Isometric-style camera angle
  - Custom blue-to-pink gradient (#19448e → #f4b3c2) optimized for clarity
- **Performance optimized** with configurable sampling
- Automatically generated as part of comprehensive report
- Saved as `expert_activation_3d.html`
- **Documentation:**
  - `COLOR_SCHEME.md`: Detailed color scheme explanation and customization guide

#### Color Scheme Improvements
- **Custom gradient** replacing default "Hot" colorscale
- **5-point gradient** for smooth color transitions:
  - 0%: #19448e (deep blue) - low activation
  - 25%: #4a6fa5 (medium blue)
  - 50%: #7d8db8 (light blue-purple)
  - 75%: #c49fbb (light purple-pink)
  - 100%: #f4b3c2 (pink) - high activation
- **Benefits:**
  - Avoids white color that was hard to see on light backgrounds
  - Better contrast and data differentiation
  - Colorblind-friendly (excellent for red-green colorblindness)
  - Smooth and natural color progression

### Added - 2025-10-30

#### Model Caching (CPU Mode)
- **Cache converted float32 models** for 5-10 minutes faster subsequent runs
- `--cache_dir PATH`: Specify cache directory to save/load converted model
- `--dump_cache`: Save converted model after running analysis
- `--dump_only`: Only convert and cache, then exit (fastest cache creation)
- **Features:**
  - Automatic cache detection and loading
  - SafeTensors format for cross-platform compatibility
  - One-time conversion, reuse for all future runs
  - Saves ~400-500 GB for MiniMax-M2 float32 cache
- **Documentation:**
  - `CACHING.md`: Comprehensive caching guide with examples
  - Updated README and USAGE with cache instructions
  
#### Command-line Arguments Support
- **Prompt Configuration**
  - `--prompt`: Specify input prompt via command-line or text file
  - `--max_tokens`: Customize maximum tokens to generate
  - Auto-detection of prompt files vs. text strings
  - UTF-8 encoding support for international characters
  
#### Analysis Control Parameters
- **Expert Weight Similarity**
  - `--enable_expert_similarity`: Toggle expert weight similarity computation
  - `--n_jobs`: Control parallelization (auto-detect or manual specification)
  - Default: disabled for faster execution
  
#### Output Configuration
- **Structured Data Export**
  - `--disable_structured_output`: Toggle JSON data export
  - `--output_format`: Choose between json, jsonl, or pickle
  - `--output_dir`: Specify custom output directory
  - Default: JSON format enabled with auto-generated timestamped directories
  
#### Documentation
- `USAGE.md`: Comprehensive command-line usage guide
- `QUICK_START.md`: Quick start guide for beginners
- `STRUCTURED_DATA.md`: JSON data format documentation
- `example_prompt.txt`: Example prompt file
- `run_examples.sh`: Shell script with usage examples

#### Features
- **Flexible Prompt Input**
  - Direct string via command-line
  - Load from text file (auto-detection)
  - Supports long prompts via files
  - UTF-8 encoding for all languages
  
- **Performance Optimizations**
  - Parallelized expert weight similarity computation (using multiprocessing)
  - Configurable thread count for high-core-count systems
  - Optional expert similarity (disabled by default for speed)
  
- **Structured Data Export**
  - JSON format for easy parsing
  - JSONL format for streaming/line-by-line processing
  - Pickle format for Python object preservation
  - Separate summary JSON for quick overview
  - NaN-safe serialization
  
- **Help System**
  - Comprehensive `--help` documentation
  - Usage examples in help text
  - Parameter descriptions

### Changed

#### `test_minimax_m2.py`
- Converted from hardcoded configuration to command-line argument parser
- Added `parse_arguments()` function
- Added `load_prompt()` function for flexible prompt loading
- Default values remain the same for backward compatibility
- Enhanced output display (truncates long prompts)

#### `moe_analyzer.py`
- Added parallelization support to `compute_expert_weight_similarity()`
- New parameters: `use_parallel` and `n_jobs`
- Added `_compute_similarities_serial()` (original method)
- Added `_compute_similarities_parallel()` (new parallel method)
- Added `_extract_expert_weights()` helper function
- Added global `_compute_similarity_worker()` for multiprocessing

#### `visualizer.py`
- Added `export_structured_data()` method
- Added `_convert_nan_to_none()` helper for JSON serialization
- Supports json, jsonl, and pickle export formats
- Exports comprehensive analysis data
- Exports compact summary JSON
- Modified `create_comprehensive_report()` to conditionally call expert similarity computation

#### `README.md`
- Added dedicated section for `test_minimax_m2.py` command-line options
- Reorganized command-line options documentation
- Added examples for new parameters
- Updated quick start guide

### Fixed

#### CPU Compatibility
- FP8 quantization handling for CPU execution
- Explicit float32 conversion after model loading
- Environment variable setup before torch import
- Device map configuration for CPU-only systems

#### Import Compatibility
- Try-except blocks for relative/absolute imports
- Support for both package and direct script execution

#### Windows Compatibility
- Removed `chmod` dependency (not needed on Windows)
- Shell scripts work with Git Bash / WSL

## [Initial Release]

### Added

#### Core Features
- MoE expert activation probability heatmap generation
- Layer-to-layer correlation analysis
- Periodic pattern detection (Δ=12, 24, 36)
- Router weight similarity analysis
- Expert weight similarity analysis

#### Model Support
- MiniMax (e.g., MiniMax-M2)
- DeepSeek-V3 and DeepSeek-V3.2
- Qwen3 MoE variants (e.g., Qwen3-235B-A22B)

#### Visualization
- Interactive Plotly visualizations
- Expert activation heatmaps
- Layer correlation matrices
- Periodic pattern bar charts and box plots
- Router similarity matrices
- Expert weight similarity scatter plots

#### Documentation
- README.md with comprehensive usage instructions
- CPU_USAGE.md for CPU-specific guidance
- Example usage scripts
- Requirements.txt with dependencies

#### Scripts
- `analyze_model.py`: Generic analyzer for all supported models
- `test_minimax_m2.py`: Dedicated test script for MiniMax-M2
- `example_usage.py`: Programmatic usage example

#### Core Modules
- `moe_hooks.py`: Hook management for capturing activations
- `moe_analyzer.py`: Statistical analysis and correlation computation
- `visualizer.py`: Plotly-based visualization generation

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| Unreleased | 2025-10-30 | Command-line arguments, parallelization, structured export |
| Initial | 2025-10-30 | First implementation with core features |

## Upgrade Notes

### From Initial to Unreleased

**Breaking Changes:**
- None. All changes are backward compatible.

**New Features:**
- Command-line arguments: Can now customize prompt, max_tokens, and analysis options via CLI
- Structured data export: Analysis results available as JSON/JSONL/Pickle
- Parallelization: Expert similarity computation now uses multiprocessing

**Migration:**
- Old usage still works (hardcoded values in script)
- New usage: add command-line arguments for customization
- To enable expert similarity: add `--enable_expert_similarity` flag
- To export different format: add `--output_format jsonl` or `--output_format pickle`

**Performance:**
- Expert similarity computation now 10-50x faster with parallelization
- Default behavior unchanged (expert similarity disabled by default)

## Future Roadmap

### Planned Features
- [ ] Support for additional MoE models (Mixtral, etc.)
- [ ] Real-time visualization during generation
- [ ] Expert activation dashboard
- [ ] Comparative analysis across multiple runs
- [ ] Token-level expert routing visualization
- [ ] Export to CSV/Excel formats
- [ ] Integration with TensorBoard
- [ ] Streaming analysis for very long sequences
- [ ] GPU parallelization for expert similarity
- [ ] Interactive web UI

### Under Consideration
- [ ] Support for non-MoE models (for comparison)
- [ ] Time-series analysis of expert activation
- [ ] Cluster analysis of expert specialization
- [ ] Attention pattern correlation with expert activation
- [ ] Fine-tuning impact analysis

