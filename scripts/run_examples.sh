#!/bin/bash
# Example usage scripts for MiniMax-M2 MoE Expert Activation Analysis

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "MiniMax-M2 Analysis Examples"
echo "=========================================="

# Example 1: Quick test with default settings
echo -e "\n[Example 1] Quick test with default settings"
echo "Command: python test_minimax_m2.py"
echo "Expected time: ~5-10 minutes"
echo ""
# Uncomment to run:
# python test_minimax_m2.py

# Example 2: Custom prompt as argument
echo -e "\n[Example 2] Custom prompt as command-line argument"
echo 'Command: python test_minimax_m2.py --prompt "Write a Python function to calculate the fibonacci sequence"'
echo "Expected time: ~5-10 minutes"
echo ""
# Uncomment to run:
# python test_minimax_m2.py --prompt "Write a Python function to calculate the fibonacci sequence"

# Example 3: Load prompt from file
echo -e "\n[Example 3] Load prompt from text file"
echo "Command: python test_minimax_m2.py --prompt ../examples/example_prompt.txt --max_tokens 1024"
echo "Expected time: ~10-15 minutes (generates more tokens)"
echo ""
# Uncomment to run:
# python test_minimax_m2.py --prompt ../examples/example_prompt.txt --max_tokens 1024

# Example 4: Enable expert similarity analysis with parallel processing
echo -e "\n[Example 4] Full analysis with expert weight similarity (parallel)"
echo "Command: python test_minimax_m2.py --enable_expert_similarity --n_jobs 64"
echo "Expected time: ~15-20 minutes (includes expert similarity computation)"
echo ""
# Uncomment to run:
# python test_minimax_m2.py --enable_expert_similarity --n_jobs 64

# Example 5: Custom output directory and format
echo -e "\n[Example 5] Custom output directory and JSONL format"
echo "Command: python test_minimax_m2.py --output_dir ./my_results --output_format jsonl"
echo "Expected time: ~5-10 minutes"
echo ""
# Uncomment to run:
# python test_minimax_m2.py --output_dir ./my_results --output_format jsonl

# Example 6: Full customization
echo -e "\n[Example 6] Full customization"
cat << 'EOF'
Command: python test_minimax_m2.py \
  --prompt ../examples/example_prompt.txt \
  --max_tokens 1024 \
  --enable_expert_similarity \
  --n_jobs 100 \
  --output_format json \
  --output_dir ./results/comprehensive_analysis
EOF
echo "Expected time: ~20-30 minutes (full analysis with high token count)"
echo ""
# Uncomment to run:
# python test_minimax_m2.py \
#   --prompt ../examples/example_prompt.txt \
#   --max_tokens 1024 \
#   --enable_expert_similarity \
#   --n_jobs 100 \
#   --output_format json \
#   --output_dir ./results/comprehensive_analysis

# Example 7: Batch processing with different prompts
echo -e "\n[Example 7] Batch processing multiple prompts"
cat << 'EOF'
# Create prompt files
echo "Explain quantum computing" > prompt1.txt
echo "Explain machine learning" > prompt2.txt
echo "Explain neural networks" > prompt3.txt

# Process each prompt
for i in {1..3}; do
  python test_minimax_m2.py \
    --prompt prompt${i}.txt \
    --output_dir ./results/batch_${i}
done
EOF
echo "Expected time: ~15-30 minutes (3 separate runs)"
echo ""

# Example 8: View help
echo -e "\n[Example 8] View all available options"
echo "Command: python test_minimax_m2.py --help"
echo ""
# Uncomment to run:
# python test_minimax_m2.py --help

echo "=========================================="
echo "To run any example:"
echo "1. Uncomment the corresponding line in this script"
echo "2. Or copy the command and run it directly"
echo "=========================================="

