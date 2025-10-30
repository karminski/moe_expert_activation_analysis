"""
MiniMax-M2 专用测试脚本

用于测试MiniMax-M2模型的MoE专家激活分析
"""

import os

# ⚠️ 重要：必须在导入torch之前设置环境变量
# 强制使用CPU模式，禁用CUDA
USE_CPU_MODE = True  # 设置为False以使用GPU

if USE_CPU_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# Import analyzer modules
from moe_analyzer import MoEAnalyzer
from visualizer import MoEVisualizer


def main():
    """MiniMax-M2 专用测试函数"""

    # ==================== 配置参数 ====================
    MODEL_PATH = "/hc550x10rz2-01/llms/MiniMax/MiniMax-M2"
    MODEL_TYPE = "minimax"
    PROMPT = "Please help me write a Python program to render an ASCII character set of the Mandelbrot set"
    MAX_LENGTH = 512
    OUTPUT_DIR = f"./minimax_m2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    PERIODIC_INTERVALS = [12, 24, 36]  # 检测Δ=12, 24, 36的周期性模式

    # CPU运行配置（从文件开头的USE_CPU_MODE变量读取）
    USE_CPU = USE_CPU_MODE
    DEVICE = "cpu" if USE_CPU else "auto"
    DTYPE = torch.float32 if USE_CPU else torch.bfloat16  # CPU使用float32

    print("\n" + "=" * 70)
    print("MiniMax-M2 MoE Expert Activation Analysis")
    print("=" * 70)
    print(f"\n📍 Model Path: {MODEL_PATH}")
    print(f"📝 Prompt: {PROMPT}")
    print(f"📊 Max Length: {MAX_LENGTH}")
    print(f"🔍 Periodic Intervals: {PERIODIC_INTERVALS}")
    print(f"💾 Output Directory: {OUTPUT_DIR}")
    print(f"🖥️  Device: {DEVICE}")
    print(f"🔢 Dtype: {DTYPE}")

    if USE_CPU:
        print("\n⚠️  CPU Mode Enabled (CUDA disabled via environment variable)")
        print("    Note: CPU inference will be significantly slower than GPU.")
        print("    For large models, this may take considerable time.")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        print(f"    CUDA device count: {torch.cuda.device_count()}")

    # ==================== 加载模型 ====================
    print("\n" + "-" * 70)
    print("[1/5] Loading Model and Tokenizer...")
    print("-" * 70)

    try:
        print("\n🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")

        print("\n🔄 Loading model (this may take a while)...")

        if USE_CPU:
            # CPU模式：直接加载到CPU，使用float32
            print("    Loading configuration...")
            from transformers import AutoConfig

            # 加载配置但不使用量化
            config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

            # 删除量化配置属性（如果存在）
            if hasattr(config, "quantization_config"):
                delattr(config, "quantization_config")

            print("    Configuration loaded (quantization disabled)")
            print("    Loading model weights...")

            # 使用修改后的配置加载模型
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                config=config,
                dtype=DTYPE,  # 使用dtype而非torch_dtype
                device_map={"": "cpu"},  # 强制使用CPU
                low_cpu_mem_usage=True,  # 减少CPU内存使用
                trust_remote_code=True,
            )
        else:
            # GPU模式：使用device_map自动分配
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=DTYPE,  # 使用dtype而非torch_dtype
                device_map=DEVICE,
                trust_remote_code=True,
            )

        print("✅ Model loaded successfully")

        # 强制转换所有参数和缓冲区到float32（解决FP8混合问题）
        if USE_CPU:
            print("\n🔄 Converting all weights to float32...")
            model = model.float()  # 转换所有参数

            # 确保所有缓冲区也是float32
            for name, buffer in model.named_buffers():
                if buffer.dtype != torch.float32:
                    buffer.data = buffer.data.float()

            print("✅ All weights converted to float32")

        # 打印模型信息
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        print(f"\n📌 Model Device: {device}")
        print(f"📌 Model Dtype: {dtype}")

        # 统计MoE层数量
        moe_layer_count = 0
        total_layers = len(model.model.layers)
        for layer in model.model.layers:
            if hasattr(layer, "block_sparse_moe"):
                moe_layer_count += 1
        print(f"📌 Total Layers: {total_layers}")
        print(f"📌 MoE Layers: {moe_layer_count}")

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\n💡 Tips:")
        print("  - Check if the model path is correct")
        print("  - Make sure you have enough GPU memory")
        print("  - Try using torch_dtype=torch.float16 if bfloat16 is not supported")
        return

    # ==================== 初始化分析器 ====================
    print("\n" + "-" * 70)
    print("[2/5] Initializing MoE Analyzer...")
    print("-" * 70)

    analyzer = MoEAnalyzer(model, model_type=MODEL_TYPE)
    print(f"✅ Analyzer initialized")
    print(f"📌 Detected Model Type: {analyzer.model_type}")

    # ==================== 准备输入 ====================
    print("\n" + "-" * 70)
    print("[3/5] Preparing Input and Running Generation...")
    print("-" * 70)

    print(f"\n📝 Tokenizing prompt...")
    inputs = tokenizer(PROMPT, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_length = inputs["input_ids"].shape[1]
    print(f"✅ Input tokens: {input_length}")

    # ==================== 运行生成并记录激活 ====================
    print(f"\n🚀 Generating (max_length={MAX_LENGTH})...")
    print("⏱️  This will take a while, please wait...")

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

        output_length = outputs.shape[1]
        generated_length = output_length - input_length

        print(f"\n✅ Generation completed!")
        print(f"📌 Total tokens: {output_length}")
        print(f"📌 Generated tokens: {generated_length}")

        # 解码生成的文本
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("\n" + "=" * 70)
        print("Generated Text:")
        print("=" * 70)
        print(generated_text)
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        print("\n💡 Tips:")
        print("  - Try reducing max_length")
        print("  - Check GPU memory usage")
        return

    # ==================== 获取统计摘要 ====================
    print("\n" + "-" * 70)
    print("[4/5] Processing Collected Data...")
    print("-" * 70)

    summary = analyzer.get_summary_statistics()
    print(f"\n✅ Data processing completed!")
    print(f"📌 MoE Layers Analyzed: {summary['num_moe_layers']}")
    print(f"📌 Layer Indices: {summary['layer_indices']}")
    print(f"📌 Total Tokens Analyzed: {summary['total_tokens_analyzed']}")

    if summary["num_experts_per_layer"]:
        first_layer = list(summary["num_experts_per_layer"].keys())[0]
        num_experts = summary["num_experts_per_layer"][first_layer]
        top_k = summary["top_k_per_layer"][first_layer]
        print(f"📌 Experts per Layer: {num_experts}")
        print(f"📌 Top-K per Token: {top_k}")

    # ==================== 生成可视化 ====================
    print("\n" + "-" * 70)
    print("[5/5] Generating Visualizations and Reports...")
    print("-" * 70)

    visualizer = MoEVisualizer()

    try:
        visualizer.create_comprehensive_report(
            analyzer, output_dir=OUTPUT_DIR, periodic_intervals=PERIODIC_INTERVALS
        )

        print("\n" + "=" * 70)
        print("✅ Analysis Complete!")
        print("=" * 70)

        print(f"\n📂 Results saved to: {OUTPUT_DIR}")
        print("\n📊 Generated files:")

        files = [
            ("expert_activation_heatmap.html", "专家激活热度图"),
            ("layer_correlation_matrix.html", "层间相关性矩阵"),
            ("periodic_pattern_analysis.html", "周期性模式分析"),
            ("periodic_pattern_analysis_detailed.html", "周期性模式详细分析"),
            ("router_similarity_matrix.html", "路由器相似度矩阵"),
            ("router_similarity_matrix_column_norms.html", "路由器列范数相关性"),
            ("expert_weight_similarity.html", "专家权重相似度分布"),
            ("expert_weight_similarity_scatter.html", "专家权重相似度散点图"),
            ("summary_report.txt", "统计摘要报告"),
        ]

        for filename, description in files:
            filepath = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                print(f"  ✅ {filename:<45} ({size_kb:.1f} KB) - {description}")

        print("\n" + "=" * 70)
        print("🎉 All Done!")
        print("=" * 70)
        print("\n💡 Next Steps:")
        print(f"  1. Open the HTML files in your browser to explore the visualizations")
        print(
            f"  2. Check {os.path.join(OUTPUT_DIR, 'summary_report.txt')} for statistics"
        )
        print(f"  3. Look for bright bands at Δ=12 in the correlation matrix!")

        print("\n🔍 Key Things to Look For:")
        print("  • Are there periodic patterns at Δ=12, 24, or 36?")
        print("  • Which experts are most frequently activated?")
        print(
            "  • Is there high similarity between routers at specific layer distances?"
        )
        print(
            "  • Do expert weights show patterns suggesting layer stacking/upcycling?"
        )

    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print(
        """
    ╔════════════════════════════════════════════════════════════════════╗
    ║         MiniMax-M2 MoE Expert Activation Analysis Tool             ║
    ║                                                                    ║
    ║  This script analyzes expert activation patterns in MiniMax-M2    ║
    ║  and generates interactive visualizations to explore:              ║
    ║  • Expert activation probabilities                                 ║
    ║  • Layer-to-layer correlations                                     ║
    ║  • Periodic patterns (Δ=12, 24, 36...)                           ║
    ║  • Router and expert weight similarities                           ║
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
