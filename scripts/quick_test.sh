#!/bin/bash
# Quick test script to verify the fix for early generation stopping

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Quick Generation Test"
echo "=========================================="
echo ""
echo "This script tests different generation configurations"
echo "to find the most stable one for your system."
echo ""

# Test 1: Greedy decoding with short generation
echo "[Test 1/4] Greedy decoding, 128 tokens"
echo "Command: python test_minimax_m2.py --prompt 'Write a hello world' --max_tokens 128 --no_sample"
echo "Expected: Should generate ~128 tokens"
echo ""
read -p "Press Enter to run test 1 (or Ctrl+C to skip)..."
python test_minimax_m2.py \
  --prompt "Write a hello world program in Python" \
  --max_tokens 128 \
  --no_sample \
  --disable_structured_output

echo ""
echo "Check: Did it generate close to 128 tokens?"
echo ""
read -p "Press Enter to continue..."

# Test 2: Greedy decoding with longer generation
echo ""
echo "[Test 2/4] Greedy decoding, 512 tokens"
echo "Command: python test_minimax_m2.py --prompt 'Write a fibonacci function' --max_tokens 512 --no_sample"
echo "Expected: Should generate ~512 tokens"
echo ""
read -p "Press Enter to run test 2 (or Ctrl+C to skip)..."
python test_minimax_m2.py \
  --prompt "Write a Python function to calculate fibonacci numbers with detailed comments" \
  --max_tokens 512 \
  --no_sample \
  --disable_structured_output

echo ""
echo "Check: Did it generate close to 512 tokens?"
echo ""
read -p "Press Enter to continue..."

# Test 3: Low temperature sampling
echo ""
echo "[Test 3/4] Low temperature sampling, 256 tokens"
echo "Command: python test_minimax_m2.py --temperature 0.3 --max_tokens 256"
echo "Expected: Should generate ~256 tokens with sampling"
echo ""
read -p "Press Enter to run test 3 (or Ctrl+C to skip)..."
python test_minimax_m2.py \
  --prompt "Explain how quicksort works" \
  --max_tokens 256 \
  --temperature 0.3 \
  --top_p 0.95 \
  --disable_structured_output

echo ""
echo "Check: Did it generate close to 256 tokens?"
echo ""
read -p "Press Enter to continue..."

# Test 4: Default sampling
echo ""
echo "[Test 4/4] Default sampling (temperature=0.7), 256 tokens"
echo "Command: python test_minimax_m2.py --max_tokens 256"
echo "Expected: Should generate ~256 tokens"
echo ""
read -p "Press Enter to run test 4 (or Ctrl+C to skip)..."
python test_minimax_m2.py \
  --prompt "Write a Python function to sort a list" \
  --max_tokens 256 \
  --disable_structured_output

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Based on your results:"
echo ""
echo "✅ If Test 1-2 worked well (greedy decoding):"
echo "   → Use --no_sample for reliable generation"
echo "   → Recommended: python test_minimax_m2.py --no_sample"
echo ""
echo "✅ If Test 3 worked well (low temperature):"
echo "   → Use --temperature 0.3 for more stable sampling"
echo "   → Recommended: python test_minimax_m2.py --temperature 0.3"
echo ""
echo "✅ If Test 4 worked well (default):"
echo "   → Default settings work fine on your system"
echo "   → No special flags needed"
echo ""
echo "❌ If all tests generated very few tokens:"
echo "   → This might be an issue with the FP8→float32 conversion"
echo "   → Try using a GPU if available"
echo "   → Check TROUBLESHOOTING.md for more solutions"
echo ""
echo "=========================================="

