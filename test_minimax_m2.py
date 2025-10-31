"""
MiniMax-M2 专用测试脚本

用于测试MiniMax-M2模型的MoE专家激活分析
"""

import os
import argparse

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MiniMax-M2 MoE Expert Activation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default prompt
  python test_minimax_m2.py
  
  # Specify custom prompt
  python test_minimax_m2.py --prompt "Write a Python function to calculate fibonacci"
  
  # Load prompt from file
  python test_minimax_m2.py --prompt prompt.txt --max_tokens 1024
  
  # Enable expert weight similarity analysis
  python test_minimax_m2.py --enable_expert_similarity --n_jobs 64
  
  # Cache the float32 converted model for faster future runs
  python test_minimax_m2.py --cache_dir ./model_cache --dump_cache
  
  # Use cached model (skip conversion)
  python test_minimax_m2.py --cache_dir ./model_cache
  
  # Only convert and cache, don't run generation
  python test_minimax_m2.py --cache_dir ./model_cache --dump_only
        """,
    )

    # Prompt configuration
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation. Can be a text string or path to a text file.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum number of NEW tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )

    # Analysis configuration
    parser.add_argument(
        "--enable_expert_similarity",
        action="store_true",
        help="Enable expert weight similarity computation (time-consuming)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for expert similarity (default: auto)",
    )
    parser.add_argument(
        "--disable_structured_output",
        action="store_true",
        help="Disable structured data output (JSON)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["json", "jsonl", "pickle"],
        default="json",
        help="Structured output format (default: json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated with timestamp)",
    )

    # Model cache configuration
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to save/load cached float32 model (for CPU mode)",
    )
    parser.add_argument(
        "--dump_cache",
        action="store_true",
        help="Save the converted float32 model to cache_dir after conversion",
    )
    parser.add_argument(
        "--dump_only",
        action="store_true",
        help="Only convert and cache the model, then exit (no generation)",
    )

    return parser.parse_args()


def load_prompt(prompt_arg):
    """
    Load prompt from argument or file.

    Args:
        prompt_arg: Prompt string or file path

    Returns:
        Prompt text
    """
    if prompt_arg is None:
        return None

    # Check if it's a file path
    if os.path.isfile(prompt_arg):
        print(f"📄 Loading prompt from file: {prompt_arg}")
        try:
            with open(prompt_arg, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
            print(f"✓ Loaded {len(prompt_text)} characters from file")
            return prompt_text
        except Exception as e:
            print(f"⚠️  Error reading file: {e}")
            print(f"   Using argument as prompt text instead")
            return prompt_arg
    else:
        # It's a direct prompt string
        return prompt_arg


def main():
    """MiniMax-M2 专用测试函数"""

    # Parse command line arguments
    args = parse_arguments()

    # ==================== 配置参数 ====================
    MODEL_PATH = "/hc550x10rz2-01/llms/MiniMax/MiniMax-M2"
    MODEL_TYPE = "minimax"

    # Prompt configuration (from args or default)
    DEFAULT_PROMPT = "Please help me write a Python program to render an ASCII character set of the Mandelbrot set"
    PROMPT = load_prompt(args.prompt) if args.prompt else DEFAULT_PROMPT

    # Generation configuration (from args or default)
    MAX_LENGTH = args.max_tokens if args.max_tokens else 512
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    DO_SAMPLE = not args.no_sample

    # Output directory (from args or auto-generated)
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = f"./minimax_m2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    PERIODIC_INTERVALS = [12, 24, 36]  # 检测Δ=12, 24, 36的周期性模式

    # CPU运行配置（从文件开头的USE_CPU_MODE变量读取）
    USE_CPU = USE_CPU_MODE
    DEVICE = "cpu" if USE_CPU else "auto"
    DTYPE = torch.float32 if USE_CPU else torch.bfloat16  # CPU使用float32

    # ==================== 分析配置参数 ====================
    # Expert Weight Similarity 计算配置 (from args or default)
    ENABLE_EXPERT_WEIGHT_SIMILARITY = args.enable_expert_similarity
    EXPERT_SIMILARITY_N_JOBS = args.n_jobs

    # 结构化数据输出配置 (from args or default)
    ENABLE_STRUCTURED_OUTPUT = not args.disable_structured_output
    STRUCTURED_OUTPUT_FORMAT = args.output_format

    # Model cache configuration (from args)
    CACHE_DIR = args.cache_dir
    DUMP_CACHE = args.dump_cache or args.dump_only
    DUMP_ONLY = args.dump_only

    # 向后兼容（保留旧变量名）
    SKIP_EXPERT_WEIGHT_SIMILARITY = not ENABLE_EXPERT_WEIGHT_SIMILARITY

    print("\n" + "=" * 70)
    print("MiniMax-M2 MoE Expert Activation Analysis")
    print("=" * 70)
    print(f"\n📍 Model Path: {MODEL_PATH}")

    # Display prompt (truncate if too long)
    if len(PROMPT) > 100:
        prompt_display = PROMPT[:97] + "..."
        print(f"📝 Prompt: {prompt_display}")
        print(f"   (Full length: {len(PROMPT)} characters)")
    else:
        print(f"📝 Prompt: {PROMPT}")

    print(f"📊 Max New Tokens: {MAX_LENGTH}")
    print(f"🌡️  Temperature: {TEMPERATURE}")
    print(f"🎲 Top-p: {TOP_P}")
    print(f"🎯 Sampling: {'Enabled' if DO_SAMPLE else 'Disabled (Greedy)'}")
    print(f"🔍 Periodic Intervals: {PERIODIC_INTERVALS}")
    print(f"💾 Output Directory: {OUTPUT_DIR}")
    print(f"🖥️  Device: {DEVICE}")
    print(f"🔢 Dtype: {DTYPE}")
    print(
        f"📊 Expert Weight Similarity: {'Enabled' if ENABLE_EXPERT_WEIGHT_SIMILARITY else 'Disabled'}"
    )
    if ENABLE_EXPERT_WEIGHT_SIMILARITY:
        if EXPERT_SIMILARITY_N_JOBS is None:
            print(f"    Parallel jobs: Auto (all CPU cores, max 32)")
        else:
            print(f"    Parallel jobs: {EXPERT_SIMILARITY_N_JOBS}")
    print(
        f"📄 Structured Output: {'Enabled' if ENABLE_STRUCTURED_OUTPUT else 'Disabled'} (format: {STRUCTURED_OUTPUT_FORMAT})"
    )

    # Cache configuration
    if CACHE_DIR:
        print(f"💾 Model Cache: {CACHE_DIR}")
        if DUMP_ONLY:
            print("    Mode: Dump-only (convert and save, then exit)")
        elif DUMP_CACHE:
            print("    Mode: Run analysis and save cache")
        else:
            print("    Mode: Load from cache (if exists)")
    else:
        print("💾 Model Cache: Disabled (will convert every time)")

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

        # Check if we should load from cache
        load_from_cache = False
        if USE_CPU and CACHE_DIR and os.path.exists(CACHE_DIR):
            # Check for essential files
            # For large models, check for sharded format (model.safetensors.index.json)
            # For small models, check for single file (model.safetensors)
            config_exists = os.path.exists(os.path.join(CACHE_DIR, "config.json"))
            single_model_exists = os.path.exists(
                os.path.join(CACHE_DIR, "model.safetensors")
            )
            sharded_model_exists = os.path.exists(
                os.path.join(CACHE_DIR, "model.safetensors.index.json")
            )

            if config_exists and (single_model_exists or sharded_model_exists):
                load_from_cache = True
                if sharded_model_exists:
                    print(f"\n✨ Found cached float32 model at: {CACHE_DIR}")
                    print("   (Sharded format detected)")
                else:
                    print(f"\n✨ Found cached float32 model at: {CACHE_DIR}")
                print("   Loading from cache (skipping FP8→float32 conversion)...")
            else:
                print(f"\n⚠️  Cache directory exists but incomplete: {CACHE_DIR}")
                if not config_exists:
                    print("   Missing: config.json")
                if not single_model_exists and not sharded_model_exists:
                    print(
                        "   Missing: model.safetensors or model.safetensors.index.json"
                    )
                print("   Will perform conversion and save to cache.")

        if load_from_cache:
            # Load directly from cache
            print("\n🔄 Loading cached float32 model...")
            model = AutoModelForCausalLM.from_pretrained(
                CACHE_DIR,
                dtype=DTYPE,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            print("✅ Cached model loaded successfully")
            print("   ⚡ Skipped FP8→float32 conversion (using cached version)")
        elif USE_CPU:
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
        if USE_CPU and not load_from_cache:
            print("\n🔄 Converting all weights to float32...")
            print("    ⚠️  Warning: Converting FP8 quantized model to float32")
            print("    This may cause some quality degradation in generation")
            model = model.float()  # 转换所有参数

            # 确保所有缓冲区也是float32
            for name, buffer in model.named_buffers():
                if buffer.dtype != torch.float32:
                    buffer.data = buffer.data.float()

            print("✅ All weights converted to float32")
            print(
                "    Note: For production use, consider using a GPU-compatible environment"
            )

            # Save to cache if requested
            if DUMP_CACHE and CACHE_DIR:
                print(f"\n💾 Saving converted model to cache: {CACHE_DIR}")
                print(
                    "    This will take a few minutes but will save time on future runs..."
                )
                try:
                    # Create cache directory if it doesn't exist
                    os.makedirs(CACHE_DIR, exist_ok=True)

                    # Save the model
                    model.save_pretrained(
                        CACHE_DIR,
                        safe_serialization=True,  # Use safetensors format
                        max_shard_size="5GB",  # Shard if model is too large
                    )

                    # Also save the tokenizer for convenience
                    tokenizer.save_pretrained(CACHE_DIR)

                    print(f"✅ Model cached successfully to: {CACHE_DIR}")
                    print("   Next time, use --cache_dir to load instantly!")

                    # If dump_only mode, exit after saving
                    if DUMP_ONLY:
                        print("\n" + "=" * 70)
                        print(
                            "🎉 Dump-only mode: Model conversion and caching completed!"
                        )
                        print("=" * 70)
                        print(f"\n📁 Cached model location: {CACHE_DIR}")
                        print("\n💡 To use the cached model in future runs:")
                        print(f"   python test_minimax_m2.py --cache_dir {CACHE_DIR}")
                        print("\n✅ Exiting (no generation performed)")
                        return

                except Exception as e:
                    print(f"⚠️  Warning: Failed to save cache: {e}")
                    print("   Continuing with analysis...")

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
    print(f"\n🚀 Generating (max_new_tokens={MAX_LENGTH})...")
    print(
        f"   Input tokens: {input_length}, will generate up to {MAX_LENGTH} new tokens"
    )
    print("⏱️  This will take a while, please wait...")

    # Debug: Check tokenizer configuration
    print(f"\n🔍 Tokenizer info:")
    print(f"   - eos_token_id: {tokenizer.eos_token_id}")
    print(f"   - pad_token_id: {tokenizer.pad_token_id}")
    print(f"   - bos_token_id: {tokenizer.bos_token_id}")

    try:
        with analyzer.record():
            with torch.no_grad():
                # Prepare generation kwargs
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": MAX_LENGTH,
                    "pad_token_id": (
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                }

                if DO_SAMPLE:
                    gen_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": TEMPERATURE,
                            "top_p": TOP_P,
                        }
                    )
                else:
                    gen_kwargs.update(
                        {
                            "do_sample": False,
                        }
                    )

                print(f"\n⚙️  Generation parameters:")
                print(f"   - max_new_tokens: {MAX_LENGTH}")
                print(f"   - do_sample: {gen_kwargs['do_sample']}")
                if DO_SAMPLE:
                    print(f"   - temperature: {TEMPERATURE}")
                    print(f"   - top_p: {TOP_P}")
                print(f"   - pad_token_id: {gen_kwargs['pad_token_id']}")
                print()

                outputs = model.generate(**gen_kwargs)

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
        print("  - Try reducing --max_tokens")
        print("  - Try using greedy decoding: --no_sample")
        print("  - Try lowering temperature: --temperature 0.3")
        print("  - Check memory usage")
        import traceback

        traceback.print_exc()
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

    if SKIP_EXPERT_WEIGHT_SIMILARITY:
        print("\n⏭️  Note: Expert weight similarity analysis is DISABLED")
        print("   To enable: Set ENABLE_EXPERT_WEIGHT_SIMILARITY = True in the script")
        print("   To control parallelism: Set EXPERT_SIMILARITY_N_JOBS = <number>")
        print("   (Recommended: Use parallel mode for faster computation)")

    visualizer = MoEVisualizer()

    try:
        if SKIP_EXPERT_WEIGHT_SIMILARITY:
            # 手动调用各个分析步骤，跳过expert weight similarity
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            print("\n" + "=" * 60)
            print("Generating MoE Analysis Report (Fast Mode)")
            print("=" * 60)

            # 1. Expert Activation Heatmap (2D)
            print("\n[1/6] Generating expert activation heatmap (2D)...")
            activation_matrix = analyzer.get_expert_activation_matrix()
            visualizer.plot_expert_activation_heatmap(
                activation_matrix,
                layer_indices=analyzer.layer_indices,
                save_path=os.path.join(OUTPUT_DIR, "expert_activation_heatmap.html"),
            )

            # 2. Expert Activation 3D Visualization (NEW!)
            print("\n[2/6] Generating expert activation 3D visualization...")
            print("    This may take a moment for large models...")
            visualizer.plot_expert_activation_3d(
                analyzer,
                save_path=os.path.join(OUTPUT_DIR, "expert_activation_3d.html"),
                max_tokens=50,  # Limit for performance
                max_layers=10,  # Show up to 10 layers
                max_experts=64,  # Show up to 64 experts
            )

            # 3. Layer Correlation Matrix
            print("\n[3/6] Computing and plotting layer correlation matrix...")
            correlation_data = analyzer.compute_layer_correlation_matrix(delta_max=30)
            visualizer.plot_layer_correlation_matrix(
                correlation_data,
                save_path=os.path.join(OUTPUT_DIR, "layer_correlation_matrix.html"),
            )

            # 4. Periodic Patterns
            print(
                f"\n[4/6] Analyzing periodic patterns (intervals: {PERIODIC_INTERVALS})..."
            )
            periodic_data = analyzer.compute_periodic_patterns(
                intervals=PERIODIC_INTERVALS
            )
            visualizer.plot_periodic_pattern(
                periodic_data,
                save_path=os.path.join(OUTPUT_DIR, "periodic_pattern_analysis.html"),
            )

            # 5. Router Weight Similarity
            print("\n[5/6] Computing router weight similarity...")
            router_sim_data = analyzer.compute_router_weight_similarity()
            visualizer.plot_router_similarity_matrix(
                router_sim_data,
                save_path=os.path.join(OUTPUT_DIR, "router_similarity_matrix.html"),
            )

            # 6. Expert Weight Similarity (如果启用)
            if ENABLE_EXPERT_WEIGHT_SIMILARITY:
                print(
                    f"\n[6/6] Computing expert weight similarity (delta={PERIODIC_INTERVALS[0]})..."
                )
                print(
                    f"    Using parallel mode with {EXPERT_SIMILARITY_N_JOBS or 'auto'} jobs"
                )
                try:
                    expert_sim_data = analyzer.compute_expert_weight_similarity(
                        delta=PERIODIC_INTERVALS[0],
                        use_parallel=True,
                        n_jobs=EXPERT_SIMILARITY_N_JOBS,
                    )
                    visualizer.plot_expert_weight_similarity(
                        expert_sim_data,
                        save_path=os.path.join(
                            OUTPUT_DIR, "expert_weight_similarity.html"
                        ),
                    )
                    print("    ✓ Expert weight similarity computed successfully")
                except KeyboardInterrupt:
                    print("\n    ⏭️  Expert weight similarity computation interrupted")
                except Exception as e:
                    print(f"\n    ⚠️  Error in expert weight similarity: {e}")
            else:
                print(
                    "\n[6/6] Skipping expert weight similarity (ENABLE_EXPERT_WEIGHT_SIMILARITY=False)"
                )

            # Summary
            summary_final = analyzer.get_summary_statistics()
            visualizer._save_summary_report(summary_final, periodic_data, OUTPUT_DIR)

            # Export structured data (if enabled)
            if ENABLE_STRUCTURED_OUTPUT:
                print("\n" + "-" * 60)
                print("Exporting Structured Data")
                print("-" * 60)
                visualizer.export_structured_data(
                    analyzer,
                    output_dir=OUTPUT_DIR,
                    format=STRUCTURED_OUTPUT_FORMAT,
                    include_raw_data=False,  # Set to True to include raw routing probabilities
                )

            print("\n" + "=" * 60)
            print("Fast Mode Analysis complete!")
            print("=" * 60)
        else:
            visualizer.create_comprehensive_report(
                analyzer, output_dir=OUTPUT_DIR, periodic_intervals=PERIODIC_INTERVALS
            )

        print("\n" + "=" * 70)
        print("✅ Analysis Complete!")
        print("=" * 70)

        print(f"\n📂 Results saved to: {OUTPUT_DIR}")
        print("\n📊 Generated files:")

        files = [
            ("expert_activation_heatmap.html", "专家激活热度图 (2D)"),
            ("expert_activation_3d.html", "专家激活3D可视化 (交互式)"),
            ("layer_correlation_matrix.html", "层间相关性矩阵"),
            ("periodic_pattern_analysis.html", "周期性模式分析"),
            ("periodic_pattern_analysis_detailed.html", "周期性模式详细分析"),
            ("router_similarity_matrix.html", "路由器相似度矩阵"),
            ("router_similarity_matrix_column_norms.html", "路由器列范数相关性"),
            ("expert_weight_similarity.html", "专家权重相似度分布"),
            ("expert_weight_similarity_scatter.html", "专家权重相似度散点图"),
            ("summary_report.txt", "统计摘要报告"),
            (
                ("analysis_data.json", "结构化数据报告 (JSON)")
                if ENABLE_STRUCTURED_OUTPUT
                else None
            ),
            (
                ("analysis_summary.json", "分析摘要 (JSON)")
                if ENABLE_STRUCTURED_OUTPUT
                else None
            ),
        ]

        files = [f for f in files if f is not None]  # 过滤None

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
