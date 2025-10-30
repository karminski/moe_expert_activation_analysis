# MoE Expert Activation Analysis Tool

A research tool for analyzing expert activation patterns in Mixture-of-Experts (MoE) transformer models. This tool helps researchers understand how different experts are activated across layers and whether there are periodic patterns in expert selection.

## Motivation

This tool was inspired by research findings about periodic activation patterns in MoE models, particularly the observation that expert activation probabilities can show strong correlations at specific layer distances (e.g., Δ=12). See [this Twitter thread](https://x.com/kilian_maciej/status/1982612297874026731) by @kilian_maciej for the original discussion about Qwen3 models.

The tool provides comprehensive analysis including:
- Expert activation probability heatmaps
- Layer-to-layer correlation matrices
- Periodic pattern detection (e.g., Δ=12, Δ=24 patterns)
- Router weight similarity analysis
- Expert weight similarity analysis

## Supported Models

Currently supports the following MoE architectures:
- **MiniMax** (e.g., MiniMax-M2)
- **DeepSeek-V3** (and DeepSeek-V3.2)
- **Qwen3 MoE** variants (e.g., Qwen3-235B-A22B)

## Installation

### Prerequisites

```bash
# Navigate to the transformers repository root
cd transformers

# Install the tool's dependencies
pip install -r examples/research_projects/moe_expert_activation_analysis/requirements.txt

# Make sure you have transformers installed
pip install -e .
```

### Dependencies

- `plotly>=5.0.0` - For interactive visualizations
- `numpy>=1.21.0` - For numerical computations
- `scipy>=1.7.0` - For statistical analysis
- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.57.0` - Hugging Face Transformers

## Quick Start

### Basic Usage

```bash
cd examples/research_projects/moe_expert_activation_analysis

python analyze_model.py \
  --model_path "Qwen/Qwen3-235B-A22B" \
  --model_type qwen3 \
  --prompt "Once upon a time, in a distant galaxy" \
  --max_length 200 \
  --output_dir ./results/qwen3_analysis
```

### With Periodic Pattern Analysis

```bash
python analyze_model.py \
  --model_path "deepseek-ai/DeepSeek-V3" \
  --model_type deepseek_v3 \
  --prompt "Explain the concept of quantum entanglement" \
  --max_length 500 \
  --periodic_intervals 12,24,36 \
  --analyze_correlation \
  --analyze_weights \
  --output_dir ./results/deepseek_analysis
```

### For MiniMax Models

#### Using the dedicated test script (recommended)

```bash
# Basic usage with default prompt
python test_minimax_m2.py

# With custom prompt
python test_minimax_m2.py --prompt "Write a Python script to visualize the Mandelbrot set"

# Load prompt from file
python test_minimax_m2.py --prompt my_prompt.txt --max_tokens 1024

# Enable expert weight similarity analysis (time-consuming but comprehensive)
python test_minimax_m2.py --enable_expert_similarity --n_jobs 64

# Full customization
python test_minimax_m2.py \
  --prompt my_prompt.txt \
  --max_tokens 1024 \
  --enable_expert_similarity \
  --n_jobs 64 \
  --output_format json \
  --output_dir ./my_results
```

**Note:** See [USAGE.md](USAGE.md) for detailed documentation of all command-line options.

#### Using the generic analyzer

```bash
python analyze_model.py \
  --model_path "MiniMax/MiniMax-M2" \
  --model_type minimax \
  --prompt "Write a short story about artificial intelligence" \
  --max_length 300 \
  --periodic_intervals 12,24 \
  --output_dir ./results/minimax_analysis
```

## Command Line Options

### MiniMax-M2 Test Script (`test_minimax_m2.py`)

This dedicated script provides a streamlined interface for MiniMax-M2 analysis:

#### Prompt Configuration
- `--prompt TEXT_OR_FILE`: Input prompt (can be text string or path to a text file)
  - If a file path is provided, the file will be automatically read
  - Supports UTF-8 encoding for international characters
  - Default: Built-in example prompt
- `--max_tokens INT`: Maximum tokens to generate (default: 512)

#### Analysis Configuration
- `--enable_expert_similarity`: Enable expert weight similarity computation
  - **Warning**: This is time-consuming (10-30+ minutes depending on hardware)
  - Recommended to use with `--n_jobs` for parallelization
- `--n_jobs INT`: Number of parallel threads for expert similarity
  - Default: `None` (auto-detects CPU cores, max 32)
  - Recommended: 64-100 for high-core-count systems

#### Output Configuration
- `--output_dir PATH`: Custom output directory
  - Default: Auto-generated with timestamp (`minimax_m2_results_YYYYMMDD_HHMMSS`)
- `--disable_structured_output`: Disable JSON output generation
  - Default: Structured output is enabled
- `--output_format {json,jsonl,pickle}`: Structured data format
  - Default: `json`

#### Examples
```bash
# Quick test with default settings
python test_minimax_m2.py

# Custom prompt and output
python test_minimax_m2.py --prompt "Explain quantum computing" --max_tokens 256

# Load prompt from file
python test_minimax_m2.py --prompt my_research_question.txt --output_dir ./results/exp1

# Full analysis with expert similarity (recommended for research)
python test_minimax_m2.py --enable_expert_similarity --n_jobs 64 --max_tokens 1024

# View all options
python test_minimax_m2.py --help
```

### Generic Analyzer (`analyze_model.py`)

For other MoE models or more control:

#### Required Arguments

- `--model_path`: Path or name of the model on HuggingFace Hub

#### Model Arguments

- `--model_type`: Type of MoE model (`minimax`, `deepseek_v3`, `qwen3`, or `auto` for auto-detection)
- `--device`: Device to use (`cpu`, `cuda`, `cuda:0`, etc.; default: `auto`)
- `--torch_dtype`: Data type for model loading (`auto`, `float32`, `float16`, `bfloat16`; default: `auto`)
- `--trust_remote_code`: Trust remote code when loading model (flag)
- `--use_flash_attention`: Use flash attention if available (flag)

#### Input Arguments

- `--prompt`: Input prompt for generation (default: "Once upon a time, in a distant galaxy")
- `--max_length`: Maximum length for generation (default: 200)
- `--temperature`: Sampling temperature (default: 1.0)

#### Analysis Arguments

- `--output_dir`: Output directory for results (default: `./moe_analysis_results`)
- `--analyze_correlation`: Perform layer correlation analysis (flag)
- `--analyze_weights`: Perform router and expert weight similarity analysis (flag)
- `--periodic_intervals`: Comma-separated list of deltas to check for periodic patterns (default: `12,24`)
- `--delta_max`: Maximum delta for correlation matrix (default: 30)

## Output Files

The tool generates several HTML files with interactive Plotly visualizations:

### 1. `expert_activation_heatmap.html`
A heatmap showing the average activation probability for each expert in each MoE layer.
- **X-axis**: Expert index
- **Y-axis**: Layer index
- **Color**: Average activation probability

### 2. `layer_correlation_matrix.html`
A heatmap showing the correlation between expert activations at different layer distances (Δ).
- **X-axis**: Layer distance (Δ)
- **Y-axis**: Starting layer index
- **Color**: Pearson correlation coefficient
- **Bright bands at specific Δ values indicate periodic patterns**

### 3. `periodic_pattern_analysis.html`
Bar charts showing the mean correlation at specific intervals (e.g., Δ=12, Δ=24).
- Includes error bars showing standard deviation
- A detailed view with box plots is also generated

### 4. `router_similarity_matrix.html`
Heatmap of cosine similarity between router weights across layers.
- Shows which layers have similar routing behavior
- Includes column norm correlation analysis

### 5. `expert_weight_similarity.html`
Histogram showing the distribution of cosine similarities between expert weights at the specified delta.
- Shows mean and median similarity
- Scatter plot by expert index also generated

### 6. `summary_report.txt`
Text file containing summary statistics:
- Number of MoE layers
- Number of experts per layer
- Total tokens analyzed
- Periodic pattern statistics

## Programmatic Usage

You can also use the tool as a Python library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_analyzer import MoEAnalyzer
from visualizer import MoEVisualizer

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-235B-A22B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B")

# Create analyzer
analyzer = MoEAnalyzer(model, model_type="qwen3")

# Record activations during generation
inputs = tokenizer("Once upon a time", return_tensors="pt")
with analyzer.record():
    outputs = model.generate(**inputs, max_length=200)

# Get expert activation matrix
activation_matrix = analyzer.get_expert_activation_matrix()

# Analyze layer correlations
correlation_data = analyzer.compute_layer_correlation_matrix(delta_max=30)

# Check for periodic patterns
periodic_data = analyzer.compute_periodic_patterns(intervals=[12, 24])

# Generate visualizations
visualizer = MoEVisualizer()
visualizer.plot_expert_activation_heatmap(
    activation_matrix,
    layer_indices=analyzer.layer_indices,
    save_path="expert_activation.html"
)
visualizer.plot_layer_correlation_matrix(
    correlation_data,
    save_path="layer_correlation.html"
)

# Or generate a comprehensive report
visualizer.create_comprehensive_report(
    analyzer,
    output_dir="./results",
    periodic_intervals=[12, 24]
)
```

## Understanding the Results

### What to Look For

1. **Expert Activation Patterns**: 
   - Are some experts consistently more active than others?
   - Are there "dead" experts that are rarely used?

2. **Periodic Patterns**:
   - Do you see bright diagonal bands in the correlation matrix?
   - Are specific Δ values (e.g., 12, 24) showing high correlations?
   - This might indicate layer stacking or upcycling during model training

3. **Router Similarity**:
   - Are routers at different layers making similar decisions?
   - High similarity might indicate redundancy or deliberate architectural choices

4. **Expert Weight Similarity**:
   - Are experts at the same index but different layers similar?
   - This could indicate weight sharing or initialization patterns

### Interpretation

High correlations at specific layer distances (e.g., Δ=12) could indicate:
- **Depth-wise layer upcycling**: Layers initialized from other layers during training
- **Layer stacking recipes**: Architectural patterns in model construction
- **Coincidental initialization**: Same random seeds used for different layer groups

## Research Applications

This tool can be used to:
- Investigate training recipes and initialization strategies
- Detect potential model architecture patterns
- Understand expert specialization and redundancy
- Compare different MoE architectures
- Validate hypotheses about model construction

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{moe_expert_activation_analysis,
  title = {MoE Expert Activation Analysis Tool},
  author = {Hugging Face Research Projects Contributors},
  year = {2025},
  url = {https://github.com/huggingface/transformers/tree/main/examples/research_projects/moe_expert_activation_analysis}
}
```

Also consider citing the original research that inspired this tool:
- Twitter thread by Maciej Kilian: https://x.com/kilian_maciej/status/1982612297874026731

## Troubleshooting

### Memory Issues

For large models, you may run into memory issues. Try:
- Using smaller `--max_length`
- Using `--torch_dtype float16` or `--torch_dtype bfloat16`
- Analyzing shorter sequences
- Running on a machine with more GPU memory

### Model Loading Issues

If the model fails to load:
- Make sure you have the latest version of transformers: `pip install -U transformers`
- Use `--trust_remote_code` if the model requires it
- Check that you have sufficient disk space for model weights

### Visualization Issues

If HTML files don't render properly:
- Make sure you have `plotly>=5.0.0` installed
- Try opening the files in a different browser
- Check that the files were fully written (look at file sizes)

## Contributing

Contributions are welcome! If you find bugs or want to add support for more MoE architectures, please:
1. Open an issue describing the problem or feature
2. Submit a pull request with your changes
3. Ensure your code follows the existing style

## License

This tool is released under the Apache 2.0 License, same as the Transformers library.

## Acknowledgments

- Inspired by research from Maciej Kilian (@kilian_maciej)
- Built on top of Hugging Face Transformers
- Uses Plotly for interactive visualizations

