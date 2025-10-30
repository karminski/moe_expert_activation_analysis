"""
Example usage of the MoE Expert Activation Analysis Tool

This is a simple example showing how to use the tool programmatically.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from moe_analyzer import MoEAnalyzer
from visualizer import MoEVisualizer


def simple_example():
    """
    A simple example that demonstrates the basic usage.
    
    Note: This example uses a small model for demonstration.
    For production use with larger models, you may need to adjust device/dtype settings.
    """
    print("="*60)
    print("MoE Expert Activation Analysis - Simple Example")
    print("="*60)
    
    # Configuration
    # Note: Replace with actual MoE model path (e.g., "Qwen/Qwen3-235B-A22B")
    model_path = "your-moe-model-path"  # Change this!
    model_type = "auto"  # Will auto-detect
    prompt = "Once upon a time, in a distant galaxy far, far away"
    max_length = 100
    output_dir = "./example_results"
    
    print(f"\nModel: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Max length: {max_length}")
    
    # Load model and tokenizer
    print("\n[1/4] Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use FP16 to save memory
            device_map="auto"
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease update 'model_path' in this script to a valid MoE model.")
        print("Example models:")
        print("  - Qwen/Qwen3-235B-A22B")
        print("  - deepseek-ai/DeepSeek-V3")
        print("  - MiniMax/MiniMax-M2")
        return
    
    # Create analyzer
    print("\n[2/4] Initializing analyzer...")
    analyzer = MoEAnalyzer(model, model_type=model_type)
    print(f"✓ Analyzer initialized (detected model type: {analyzer.model_type})")
    
    # Run generation with recording
    print("\n[3/4] Running generation and recording activations...")
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with analyzer.record():
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Generated {outputs.shape[1]} tokens")
    print(f"\nGenerated text:\n{generated_text[:200]}...")
    
    # Get summary
    summary = analyzer.get_summary_statistics()
    print(f"\n✓ Recorded activations from {summary['num_moe_layers']} MoE layers")
    print(f"  - Total tokens: {summary['total_tokens_analyzed']}")
    print(f"  - Experts per layer: {list(summary['num_experts_per_layer'].values())[0]}")
    
    # Generate visualizations
    print("\n[4/4] Generating visualizations...")
    visualizer = MoEVisualizer()
    visualizer.create_comprehensive_report(
        analyzer,
        output_dir=output_dir,
        periodic_intervals=[12, 24]
    )
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    for filename in [
        "expert_activation_heatmap.html",
        "layer_correlation_matrix.html",
        "periodic_pattern_analysis.html",
        "router_similarity_matrix.html",
        "expert_weight_similarity.html",
        "summary_report.txt"
    ]:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")
        else:
            print(f"  - {filename}")
    
    print("\nOpen the HTML files in your web browser to view the results!")


def minimal_example():
    """
    A minimal example showing just the activation heatmap.
    """
    print("This is a minimal example. See simple_example() for a complete workflow.")
    print("\nBasic usage:")
    print("""
    from moe_analyzer import MoEAnalyzer
    from visualizer import MoEVisualizer
    
    # Load your model
    model = AutoModelForCausalLM.from_pretrained("your-moe-model")
    
    # Create analyzer and record activations
    analyzer = MoEAnalyzer(model)
    with analyzer.record():
        outputs = model.generate(inputs, max_length=100)
    
    # Get activation matrix and visualize
    activation_matrix = analyzer.get_expert_activation_matrix()
    visualizer = MoEVisualizer()
    visualizer.plot_expert_activation_heatmap(
        activation_matrix,
        layer_indices=analyzer.layer_indices,
        save_path="heatmap.html"
    )
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example usage of MoE analysis tool")
    parser.add_argument(
        "--example",
        type=str,
        default="simple",
        choices=["simple", "minimal"],
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "simple":
        simple_example()
    else:
        minimal_example()

