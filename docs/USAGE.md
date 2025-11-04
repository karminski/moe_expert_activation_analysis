# 使用指南 (Usage Guide)

## 基本用法

### 1. 默认运行
```bash
python test_minimax_m2.py
```
使用默认prompt和配置。

### 2. 自定义Prompt

#### 直接指定prompt
```bash
python test_minimax_m2.py --prompt "Write a Python function to calculate fibonacci numbers"
```

#### 从文件加载prompt
```bash
# 创建prompt文件
echo "Explain quantum computing in simple terms" > my_prompt.txt

# 使用文件
python test_minimax_m2.py --prompt my_prompt.txt
```

#### 长prompt示例
```bash
python test_minimax_m2.py --prompt example_prompt.txt --max_tokens 1024
```

### 3. 控制生成参数

#### 使用贪婪解码（推荐，更稳定）
```bash
python test_minimax_m2.py --no_sample
```
**说明**：贪婪解码在CPU模式下更稳定，避免过早停止生成。

#### 调整temperature
```bash
# 低temperature（更确定）
python test_minimax_m2.py --temperature 0.3

# 高temperature（更有创意）
python test_minimax_m2.py --temperature 1.2
```

#### 调整top_p
```bash
python test_minimax_m2.py --top_p 0.95
```

#### 组合使用
```bash
python test_minimax_m2.py \
  --prompt "Write a sorting algorithm" \
  --max_tokens 1024 \
  --temperature 0.5 \
  --top_p 0.95
```

### 4. 控制生成长度

```bash
# 生成256个新tokens
python test_minimax_m2.py --max_tokens 256

# 生成2048个新tokens  
python test_minimax_m2.py --max_tokens 2048
```

**重要**：`--max_tokens` 指定的是**新生成的token数**，不包括prompt的长度。

### 5. 启用专家权重相似度分析

```bash
# 使用自动并行
python test_minimax_m2.py --enable_expert_similarity

# 指定线程数
python test_minimax_m2.py --enable_expert_similarity --n_jobs 64

# 使用100个核心
python test_minimax_m2.py --enable_expert_similarity --n_jobs 100
```

### 6. 控制输出

#### 指定输出目录
```bash
python test_minimax_m2.py --output_dir ./my_analysis_results
```

#### 禁用结构化数据输出
```bash
python test_minimax_m2.py --disable_structured_output
```

#### 选择输出格式
```bash
# JSON格式（默认）
python test_minimax_m2.py --output_format json

# JSONL格式
python test_minimax_m2.py --output_format jsonl

# Pickle格式
python test_minimax_m2.py --output_format pickle
```

## 🚀 使用模型缓存（推荐！节省时间）

**CPU模式下必读！** 缓存可以节省每次运行5-10分钟的转换时间。

### 首次：创建缓存

```bash
# 方案1：只创建缓存（最快，推荐）
python test_minimax_m2.py --cache_dir ./model_cache --dump_only

# 方案2：运行分析并保存缓存
python test_minimax_m2.py --cache_dir ./model_cache --dump_cache
```

### 后续：使用缓存

```bash
# 所有后续运行都使用缓存（快速加载）
python test_minimax_m2.py \
  --cache_dir ./model_cache \
  --prompt "Your prompt" \
  --max_tokens 1024
```

**时间对比：**
- 不使用缓存：10-20分钟（加载+转换+生成）
- 使用缓存：6-12分钟（加载+生成）
- 节省：**4-8分钟**

详见：[CACHING.md](CACHING.md)

## 完整示例

### 示例1：快速分析（最小配置）
```bash
python test_minimax_m2.py \
  --prompt "Write a hello world program" \
  --max_tokens 128
```

### 示例2：标准分析（使用缓存）
```bash
# 首次运行：创建缓存
python test_minimax_m2.py \
  --cache_dir ./cache \
  --dump_only

# 后续运行：使用缓存
python test_minimax_m2.py \
  --cache_dir ./cache \
  --prompt my_prompt.txt \
  --max_tokens 512 \
  --output_dir ./results/test1
```

### 示例3：完整分析（包含专家权重）
```bash
python test_minimax_m2.py \
  --cache_dir ./cache \
  --prompt "Explain machine learning" \
  --max_tokens 1024 \
  --enable_expert_similarity \
  --n_jobs 64 \
  --output_dir ./results/full_analysis
```

### 示例4：批量分析（使用不同prompt）
```bash
# 一次性创建缓存
python test_minimax_m2.py --cache_dir ./cache --dump_only

# 准备多个prompt文件
echo "Explain AI" > prompt1.txt
echo "Explain ML" > prompt2.txt
echo "Explain DL" > prompt3.txt

# 快速批量处理（都使用缓存）
for i in {1..3}; do
    python test_minimax_m2.py \
      --cache_dir ./cache \
      --prompt prompt${i}.txt \
      --output_dir ./results/batch_${i}
done
```

## ⚠️ 常见问题：生成token数太少

如果你遇到生成只有几个token就停止的情况：

```
✅ Generation completed!
📌 Total tokens: 11
📌 Generated tokens: 2  ← 太少了！
```

### 解决方案

#### 方案1：使用贪婪解码（最推荐）
```bash
python test_minimax_m2.py \
  --prompt "Write a Python function to calculate fibonacci numbers" \
  --max_tokens 1024 \
  --no_sample
```

#### 方案2：降低temperature
```bash
python test_minimax_m2.py \
  --prompt "Write a Python function to calculate fibonacci numbers" \
  --max_tokens 1024 \
  --temperature 0.3
```

#### 方案3：更改prompt
```bash
# 更具体、更明确的prompt通常效果更好
python test_minimax_m2.py \
  --prompt "Below is a complete Python implementation of fibonacci:\n\ndef fibonacci(n):" \
  --max_tokens 1024 \
  --no_sample
```

查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 获取更多解决方案。

## 命令行参数完整列表

### Prompt相关
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--prompt` | str | 默认prompt | 输入prompt（文本或文件路径） |
| `--max_tokens` | int | 512 | 生成的**新**token数（不含prompt） |
| `--temperature` | float | 0.7 | 采样温度（0.1-2.0） |
| `--top_p` | float | 0.9 | Top-p采样参数 |
| `--no_sample` | flag | False | 使用贪婪解码 |

### 分析相关
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable_expert_similarity` | flag | False | 启用专家权重相似度分析 |
| `--n_jobs` | int | None | 并行线程数（None=自动） |

### 输出相关
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_dir` | str | 自动生成 | 输出目录路径 |
| `--disable_structured_output` | flag | False | 禁用JSON数据输出 |
| `--output_format` | str | json | 结构化数据格式 |

### 缓存相关（CPU模式）
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cache_dir` | str | None | 缓存目录路径 |
| `--dump_cache` | flag | False | 运行后保存缓存 |
| `--dump_only` | flag | False | 只转换和保存，不运行分析 |

## Prompt文件格式

### 纯文本
```
Write a Python function to calculate prime numbers.
```

### 多行文本
```
Please write a comprehensive guide about:
1. Data structures
2. Algorithms
3. Complexity analysis

Include code examples.
```

### UTF-8编码
文件自动使用UTF-8编码读取，支持中文和其他Unicode字符。

```
请写一个Python程序来：
1. 读取CSV文件
2. 数据清洗
3. 生成可视化图表
```

## 性能建议

### 快速测试
```bash
python test_minimax_m2.py --max_tokens 128
```
- 约2-3分钟完成
- 生成基本分析报告

### 标准分析
```bash
python test_minimax_m2.py --max_tokens 512
```
- 约5-10分钟完成
- 完整的可视化报告
- 不包含专家权重相似度

### 完整分析
```bash
python test_minimax_m2.py \
  --max_tokens 512 \
  --enable_expert_similarity \
  --n_jobs 64
```
- 约10-15分钟完成
- 所有分析项目
- 包含专家权重相似度（并行计算）

## 查看帮助

```bash
python test_minimax_m2.py --help
```

输出：
```
usage: test_minimax_m2.py [-h] [--prompt PROMPT] [--max_tokens MAX_TOKENS]
                          [--enable_expert_similarity] [--n_jobs N_JOBS]
                          [--disable_structured_output]
                          [--output_format {json,jsonl,pickle}]
                          [--output_dir OUTPUT_DIR]

MiniMax-M2 MoE Expert Activation Analysis

optional arguments:
  -h, --help            show this help message and exit
  --prompt PROMPT       Input prompt for generation. Can be a text string or
                        path to a text file.
  --max_tokens MAX_TOKENS
                        Maximum number of tokens to generate (default: 512)
  --enable_expert_similarity
                        Enable expert weight similarity computation (time-
                        consuming)
  --n_jobs N_JOBS       Number of parallel jobs for expert similarity
                        (default: auto)
  --disable_structured_output
                        Disable structured data output (JSON)
  --output_format {json,jsonl,pickle}
                        Structured output format (default: json)
  --output_dir OUTPUT_DIR
                        Output directory (default: auto-generated with
                        timestamp)
```

## 常见问题

### Q: Prompt太长会怎样？
A: 脚本会在显示时自动截断（显示前97个字符+...），但完整内容会用于生成。

### Q: 如何使用多行prompt？
A: 推荐使用prompt文件。直接在命令行输入多行prompt需要使用引号和转义：
```bash
python test_minimax_m2.py --prompt "Line 1
Line 2
Line 3"
```

### Q: 文件不存在会怎样？
A: 如果指定的文件不存在，脚本会将参数值作为prompt文本使用。

### Q: 如何查看当前配置？
A: 脚本启动时会显示所有配置参数：
```
📍 Model Path: /hc550x10rz2-01/llms/MiniMax/MiniMax-M2
📝 Prompt: Write a Python...
📊 Max Length: 512
🔍 Periodic Intervals: [12, 24, 36]
💾 Output Directory: ./minimax_m2_results_20251030_123456
🖥️  Device: cpu
🔢 Dtype: torch.float32
📊 Expert Weight Similarity: Disabled
📄 Structured Output: Enabled (format: json)
```

### Q: 如何保存配置供复用？
A: 创建一个shell脚本：
```bash
#!/bin/bash
# my_analysis.sh

python test_minimax_m2.py \
  --prompt my_prompt.txt \
  --max_tokens 1024 \
  --enable_expert_similarity \
  --n_jobs 64 \
  --output_dir ./results/$(date +%Y%m%d_%H%M%S)
```

## 高级用法

### 环境变量配置
```bash
# 设置默认输出目录
export MOE_OUTPUT_DIR="./my_results"

# 使用环境变量（需要修改脚本支持）
python test_minimax_m2.py --output_dir "$MOE_OUTPUT_DIR/run1"
```

### 与其他工具集成
```bash
# 生成后自动分析JSON
python test_minimax_m2.py --prompt test.txt && \
  python analyze_json.py ./minimax_m2_results_*/analysis_summary.json
```

### 条件执行
```bash
# 只有在文件存在时才运行
[ -f my_prompt.txt ] && python test_minimax_m2.py --prompt my_prompt.txt
```

