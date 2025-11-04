"""
MoE Expert Activation Analysis CLI

Command-line interface for analyzing MoE models.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.moe_analyzer import MoEAnalyzer
from src.visualizer import MoEVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze expert activation patterns in MoE models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Qwen3 model with default settings
  python analyze_model.py --model_path "Qwen/Qwen3-235B-A22B" --model_type qwen3
  
  # Analyze with custom prompt and check for delta=12 pattern
  python analyze_model.py \\
    --model_path "deepseek-ai/DeepSeek-V3" \\
    --model_type deepseek_v3 \\
    --prompt "Explain quantum computing" \\
    --max_length 500 \\
    --periodic_intervals 12,24,36
  
  # Run full analysis with weight similarity
  python analyze_model.py \\
    --model_path "MiniMax/MiniMax-M2" \\
    --model_type minimax \\
    --analyze_correlation \\
    --analyze_weights \\
    --output_dir ./results/minimax_analysis
"""
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or name of the model on HuggingFace Hub"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["minimax", "deepseek_v3", "qwen3", "auto"],
        help="Type of MoE model (default: auto-detect)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cpu', 'cuda', 'cuda:0', etc. (default: auto)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for model loading (default: auto)"
    )
    
    # Input arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, in a distant galaxy",
        help="Input prompt for generation (default: 'Once upon a time...')"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum length for generation (default: 200)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    
    # Analysis arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./moe_analysis_results",
        help="Output directory for results (default: ./moe_analysis_results)"
    )
    parser.add_argument(
        "--analyze_correlation",
        action="store_true",
        help="Perform layer correlation analysis"
    )
    parser.add_argument(
        "--analyze_weights",
        action="store_true",
        help="Perform router and expert weight similarity analysis"
    )
    parser.add_argument(
        "--periodic_intervals",
        type=str,
        default="12,24",
        help="Comma-separated list of deltas to check for periodic patterns (default: 12,24)"
    )
    parser.add_argument(
        "--delta_max",
        type=int,
        default=30,
        help="Maximum delta for correlation matrix (default: 30)"
    )
    
    # Model loading arguments
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model"
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use flash attention if available"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load the model and tokenizer."""
    print("\n" + "="*60)
    print("Loading Model and Tokenizer")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Model Type: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.torch_dtype}")
    
    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto"
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Determine device
    if args.device == "auto":
        device_map = "auto"
    else:
        device_map = {"": args.device}
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code
    )
    
    # Load model
    print("Loading model (this may take a while)...")
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    
    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
    )
    
    print(f"Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer


def run_generation(model, tokenizer, args):
    """Run generation with the model."""
    print("\n" + "="*60)
    print("Running Generation")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Max length: {args.max_length}")
    
    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt")
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"\nInput tokens: {inputs['input_ids'].shape[1]}")
    
    # Generate
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=args.temperature != 1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nGenerated {outputs.shape[1]} tokens")
    print("\nGenerated text:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)
    
    return outputs


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("MoE Expert Activation Analysis Tool")
    print("="*60)
    
    # Parse periodic intervals
    periodic_intervals = [int(x.strip()) for x in args.periodic_intervals.split(",")]
    print(f"\nPeriodic intervals to analyze: {periodic_intervals}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Create analyzer
    print("\n" + "="*60)
    print("Initializing Analyzer")
    print("="*60)
    analyzer = MoEAnalyzer(model, model_type=args.model_type)
    
    # Run generation with recording
    print("\n" + "="*60)
    print("Recording Expert Activations")
    print("="*60)
    
    with analyzer.record():
        outputs = run_generation(model, tokenizer, args)
    
    # Print summary
    summary = analyzer.get_summary_statistics()
    print("\n" + "="*60)
    print("Analysis Summary")
    print("="*60)
    print(f"Number of MoE Layers: {summary['num_moe_layers']}")
    print(f"Layer Indices: {summary['layer_indices']}")
    print(f"Total Tokens Analyzed: {summary['total_tokens_analyzed']}")
    print(f"Experts per Layer: {list(summary['num_experts_per_layer'].values())[0] if summary['num_experts_per_layer'] else 'N/A'}")
    
    # Create visualizer
    visualizer = MoEVisualizer()
    
    # Generate comprehensive report
    visualizer.create_comprehensive_report(
        analyzer,
        output_dir=args.output_dir,
        periodic_intervals=periodic_intervals
    )
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - expert_activation_heatmap.html")
    print("  - layer_correlation_matrix.html")
    print("  - periodic_pattern_analysis.html")
    print("  - router_similarity_matrix.html")
    print("  - expert_weight_similarity.html")
    print("  - summary_report.txt")
    print("\nOpen the HTML files in a web browser to view interactive visualizations.")
    

if __name__ == "__main__":
    main()

