# Quick Start Guide - MiniMax-M2 Analysis

这是一份快速入门指南，让你在几分钟内开始分析 MiniMax-M2 模型的专家激活模式。

## 🚀 最简单的开始

```bash
cd examples/research_projects/moe_expert_activation_analysis
python test_minimax_m2.py
```

就这么简单！脚本会：
- ✅ 使用默认prompt
- ✅ 生成512个tokens
- ✅ 创建所有可视化报告
- ✅ 生成JSON结构化数据
- ⏱️ 约5-10分钟完成（首次运行，包含FP8→float32转换）

## ⚡ 加速技巧：使用缓存（推荐！）

**如果你会多次运行，强烈推荐使用缓存！节省5-10分钟。**

```bash
# 第一次：创建缓存（只需做一次）
python test_minimax_m2.py --cache_dir ./model_cache --dump_only
# ⏱️ 约5-10分钟

# 之后每次：使用缓存（超快！）
python test_minimax_m2.py --cache_dir ./model_cache
# ⏱️ 约1-2分钟加载 + 5-10分钟生成
```

**节省时间！** 第二次运行起就快很多。详见 [CACHING.md](CACHING.md)。

## 📝 自定义Prompt

### 方法1: 命令行直接指定

```bash
python test_minimax_m2.py --prompt "Write a Python sorting algorithm"
```

### 方法2: 从文件读取

```bash
# 创建prompt文件
echo "Explain the theory of relativity in simple terms" > my_prompt.txt

# 使用文件
python test_minimax_m2.py --prompt my_prompt.txt
```

### 方法3: 使用提供的示例prompt

```bash
python test_minimax_m2.py --prompt example_prompt.txt
```

## 🎛️ 常用参数组合

### 快速测试（最快）
```bash
python test_minimax_m2.py --max_tokens 128
```
⏱️ 约2-3分钟

### 标准分析（推荐）
```bash
python test_minimax_m2.py --max_tokens 512
```
⏱️ 约5-10分钟

### 长文本分析
```bash
python test_minimax_m2.py --max_tokens 2048
```
⏱️ 约15-20分钟

### 完整分析（包含专家权重相似度）
```bash
python test_minimax_m2.py \
  --max_tokens 512 \
  --enable_expert_similarity \
  --n_jobs 64
```
⏱️ 约15-25分钟

## 📊 输出文件

运行完成后，你会得到：

### 📁 输出目录结构
```
minimax_m2_results_20251030_123456/
├── expert_activation_heatmap.html          # 专家激活热力图
├── layer_correlation_matrix.html           # 层间相关性矩阵
├── periodic_pattern_analysis.html          # 周期性模式分析
├── periodic_pattern_analysis_detailed.html # 详细周期性分析
├── router_similarity_matrix.html           # 路由器相似度矩阵
├── router_similarity_matrix_column_norms.html # 路由器列范数
├── expert_weight_similarity_matrix.html    # 专家权重相似度（如启用）
├── expert_weight_similarity_scatter.html   # 专家权重散点图（如启用）
├── summary_report.txt                      # 文本摘要报告
├── analysis_data.json                      # 结构化数据（完整）
└── analysis_summary.json                   # 结构化数据（摘要）
```

### 🌐 查看结果

在浏览器中打开任何 `.html` 文件即可：
```bash
# Windows
start minimax_m2_results_*/expert_activation_heatmap.html

# Linux/Mac
xdg-open minimax_m2_results_*/expert_activation_heatmap.html
```

### 📄 查看文本报告

```bash
cat minimax_m2_results_*/summary_report.txt
```

## 🔍 重要发现：查看周期性模式

周期性模式（Δ=12, 24, 36）是最有趣的发现！

### 在浏览器中查看
打开 `periodic_pattern_analysis.html` 和 `periodic_pattern_analysis_detailed.html`

### 在命令行中查看
```bash
# 查看摘要JSON
cat minimax_m2_results_*/analysis_summary.json | grep -A 10 "periodic_patterns"
```

### 如何解读
- **相关系数 > 0.7**: 强周期性模式（专家激活在固定层间距上高度相似）
- **相关系数 0.4-0.7**: 中等周期性模式
- **相关系数 < 0.4**: 弱或无周期性模式

## 📈 参数速查表

| 任务 | 命令 | 时间 |
|------|------|------|
| 快速测试 | `python test_minimax_m2.py --max_tokens 128` | 2-3分钟 |
| 标准分析 | `python test_minimax_m2.py` | 5-10分钟 |
| 自定义prompt | `python test_minimax_m2.py --prompt "your text"` | 5-10分钟 |
| 从文件读取 | `python test_minimax_m2.py --prompt file.txt` | 5-10分钟 |
| 长文本 | `python test_minimax_m2.py --max_tokens 2048` | 15-20分钟 |
| 完整分析 | `python test_minimax_m2.py --enable_expert_similarity --n_jobs 64` | 15-25分钟 |
| 仅网页输出 | `python test_minimax_m2.py --disable_structured_output` | 5-10分钟 |

## 🎯 推荐工作流

### 第一次使用（探索）
```bash
# 1. 快速测试确保环境正常
python test_minimax_m2.py --max_tokens 128

# 2. 查看结果
start minimax_m2_results_*/expert_activation_heatmap.html  # Windows
# or
xdg-open minimax_m2_results_*/expert_activation_heatmap.html  # Linux

# 3. 如果满意，运行标准分析
python test_minimax_m2.py --prompt "your interesting prompt"
```

### 研究用途（详细分析）
```bash
# 准备你的prompt
echo "Your research question here" > research_prompt.txt

# 运行完整分析
python test_minimax_m2.py \
  --prompt research_prompt.txt \
  --max_tokens 1024 \
  --enable_expert_similarity \
  --n_jobs 64 \
  --output_dir ./research_results/experiment_1

# 分析JSON数据
python your_analysis_script.py ./research_results/experiment_1/analysis_data.json
```

### 批量实验
```bash
# 创建多个prompt
for i in {1..5}; do
  echo "Prompt $i: [your question]" > prompt_$i.txt
done

# 批量处理
for i in {1..5}; do
  python test_minimax_m2.py \
    --prompt prompt_$i.txt \
    --output_dir ./batch_results/exp_$i
done
```

## 💡 小贴士

### 1. 查看所有可用参数
```bash
python test_minimax_m2.py --help
```

### 2. 测试不同的生成长度
生成的token数量会影响：
- **分析时间**: 更多tokens = 更长时间
- **专家激活样本**: 更多tokens = 更多样本数据 = 更可靠的统计
- **可视化质量**: 一般512-1024 tokens就足够获得清晰的模式

### 3. 是否启用专家权重相似度？
- ✅ **启用**: 研究用途，需要完整的模型分析
- ❌ **不启用**: 快速查看专家激活模式，节省时间

### 4. 并行线程数建议
- **自动（推荐）**: `--n_jobs` 不指定或设为 `None`
- **手动**: 设置为你的CPU核心数，例如 `--n_jobs 64`
- **超线程**: 可以设置为核心数的2倍试试，如 `--n_jobs 128`

### 5. 节省时间的技巧
```bash
# 使用较短的max_tokens进行初步探索
python test_minimax_m2.py --max_tokens 256

# 确认有趣的发现后再运行完整分析
python test_minimax_m2.py --max_tokens 1024 --enable_expert_similarity
```

## 🐛 遇到问题？

### 模型加载失败
检查模型路径：脚本默认使用 `/hc550x10rz2-01/llms/MiniMax/MiniMax-M2`
如需修改，编辑 `test_minimax_m2.py` 的 `MODEL_PATH` 变量。

### 内存不足
尝试：
```bash
# 减少生成长度
python test_minimax_m2.py --max_tokens 128

# 禁用专家权重相似度
python test_minimax_m2.py  # 默认就是禁用的
```

### FP8量化问题（CPU模式）
脚本已配置为自动处理FP8到float32的转换。
如有问题，查看 [CPU_USAGE.md](CPU_USAGE.md)。

### 其他问题
查看详细文档：
- [README.md](README.md) - 完整文档
- [USAGE.md](USAGE.md) - 使用指南
- [STRUCTURED_DATA.md](STRUCTURED_DATA.md) - 结构化数据说明

## 📚 更多资源

- **完整文档**: [README.md](README.md)
- **命令行参数详解**: [USAGE.md](USAGE.md)
- **JSON数据格式**: [STRUCTURED_DATA.md](STRUCTURED_DATA.md)
- **CPU运行指南**: [CPU_USAGE.md](CPU_USAGE.md)

## 🎓 下一步

1. ✅ 运行第一个分析
2. ✅ 在浏览器中查看可视化
3. ✅ 尝试不同的prompt
4. ✅ 研究周期性模式（Δ=12）
5. ✅ 分析JSON数据（编程式处理）

现在就开始吧！

```bash
python test_minimax_m2.py
```

祝你分析愉快！🚀

