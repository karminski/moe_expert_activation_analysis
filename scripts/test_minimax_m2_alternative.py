"""
MiniMax-M2 备选测试脚本（简化版）

如果主脚本遇到配置问题，可以尝试这个简化版本
"""

import os
import sys

# 强制使用CPU模式
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import analyzer modules
from src.moe_analyzer import MoEAnalyzer
from src.visualizer import MoEVisualizer


def main():
    """MiniMax-M2 简化测试函数"""

    # 配置参数
    MODEL_PATH = "/hc550x10rz2-01/llms/MiniMax/MiniMax-M2"
    PROMPT = "请帮我写一个python渲染的ASCII字符集MandelbortSet"
    MAX_LENGTH = 512
    OUTPUT_DIR = f"./minimax_m2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    PERIODIC_INTERVALS = [12, 24, 36]

    print("\n" + "=" * 70)
    print("MiniMax-M2 MoE Expert Activation Analysis (Alternative)")
    print("=" * 70)
    print(f"\n📍 Model Path: {MODEL_PATH}")
    print(f"📝 Prompt: {PROMPT}")
    print(f"🖥️  Device: CPU")
    print(f"🔢 Dtype: float32")
    print(f"\n⚠️  CPU Mode - CUDA disabled")
    print(f"    CUDA available: {torch.cuda.is_available()}")

    # 加载tokenizer
    print("\n[1/5] Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ Tokenizer loaded")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 加载模型 - 尝试最简单的方式
    print("\n[2/5] Loading Model (this will take several minutes)...")
    print("    Using simplified loading strategy...")
    
    try:
        # 方法1：最简单的加载（推荐）
        print("    Attempting simple load with explicit dtype...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully")
        
    except Exception as e1:
        print(f"    Failed: {e1}")
        print("\n    Trying alternative method...")
        
        try:
            # 方法2：不指定dtype，让它自动选择
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            # 然后转换到float32
            model = model.float()
            print("✅ Model loaded with auto dtype and converted to float32")
            
        except Exception as e2:
            print(f"    Failed: {e2}")
            print("\n❌ Unable to load model. Please check:")
            print("    1. Model path is correct")
            print("    2. Model files are not corrupted")
            print("    3. You have sufficient memory (need ~200GB+)")
            return

    # 打印模型信息
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"\n📌 Model Device: {device}")
    print(f"📌 Model Dtype: {dtype}")

    # 统计MoE层
    moe_count = sum(
        1 for layer in model.model.layers if hasattr(layer, "block_sparse_moe")
    )
    print(f"📌 Total Layers: {len(model.model.layers)}")
    print(f"📌 MoE Layers: {moe_count}")

    # 初始化分析器
    print("\n[3/5] Initializing Analyzer...")
    analyzer = MoEAnalyzer(model, model_type="minimax")
    print(f"✅ Analyzer ready")

    # 准备输入
    print("\n[4/5] Generating Text...")
    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    print(f"    Input tokens: {inputs['input_ids'].shape[1]}")

    # 生成
    print(f"    Generating up to {MAX_LENGTH} tokens (this will take a while)...")
    try:
        with analyzer.record():
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_LENGTH,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n✅ Generated {outputs.shape[1]} tokens")
        print("\n" + "=" * 70)
        print("Generated Text:")
        print("=" * 70)
        print(generated_text)
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Generation error: {e}")
        return

    # 生成分析报告
    print("\n[5/5] Generating Analysis Report...")
    summary = analyzer.get_summary_statistics()
    print(f"✅ Analyzed {summary['num_moe_layers']} MoE layers")
    print(f"    Total tokens: {summary['total_tokens_analyzed']}")

    visualizer = MoEVisualizer()
    try:
        visualizer.create_comprehensive_report(
            analyzer, output_dir=OUTPUT_DIR, periodic_intervals=PERIODIC_INTERVALS
        )
        print(f"\n🎉 Analysis complete! Results in: {OUTPUT_DIR}")
    except Exception as e:
        print(f"\n❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print(
        """
    ╔════════════════════════════════════════════════════════════════════╗
    ║      MiniMax-M2 MoE Analysis (Alternative/Simplified Version)      ║
    ╚════════════════════════════════════════════════════════════════════╝
    """
    )

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

