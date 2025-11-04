#!/bin/bash
# MiniMax-M2 CPU运行脚本
# 此脚本设置必要的环境变量以强制使用CPU

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "MiniMax-M2 CPU Analysis Runner"
echo "======================================================================"
echo ""
echo "Setting CPU-only mode..."

# 禁用CUDA设备，强制使用CPU
export CUDA_VISIBLE_DEVICES=""
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

echo "✅ CUDA disabled"
echo ""
echo "Starting analysis..."
echo ""

# 运行Python脚本
python test_minimax_m2.py

echo ""
echo "======================================================================"
echo "Analysis completed"
echo "======================================================================"

