# 模型缓存功能 (Model Caching)

## 概述

在CPU模式下，MiniMax-M2需要从FP8量化格式转换到float32，这个过程需要几分钟时间。使用模型缓存功能，你可以：

1. **一次转换，多次使用** - 将转换后的float32模型保存到磁盘
2. **秒级加载** - 后续运行直接加载缓存，跳过转换过程
3. **节省时间** - 每次运行节省5-10分钟的转换时间

## 快速开始

### 第一次运行：创建缓存

```bash
# 方案1：转换 + 运行分析 + 保存缓存
python test_minimax_m2.py \
  --cache_dir ./model_cache_float32 \
  --dump_cache

# 方案2：只转换和保存，不运行分析（更快）
python test_minimax_m2.py \
  --cache_dir ./model_cache_float32 \
  --dump_only
```

**方案2推荐**：如果你只想创建缓存供后续使用，`--dump_only` 更快。

### 后续运行：使用缓存

```bash
# 直接加载缓存的float32模型
python test_minimax_m2.py \
  --cache_dir ./model_cache_float32 \
  --prompt "Write a Python function" \
  --max_tokens 1024
```

**加载速度对比：**
- 不使用缓存：5-10分钟（加载 + 转换）
- 使用缓存：30秒-2分钟（仅加载）

## 详细说明

### 参数说明

#### `--cache_dir PATH`
指定缓存目录路径。

```bash
# 使用相对路径
--cache_dir ./model_cache

# 使用绝对路径
--cache_dir /path/to/cache/minimax_m2_float32

# 推荐的命名方式
--cache_dir ./minimax_m2_float32_cache
```

#### `--dump_cache`
运行完整分析并保存缓存。

```bash
python test_minimax_m2.py --cache_dir ./cache --dump_cache
```

**用途：**
- 第一次运行时创建缓存
- 既要分析结果，又要保存缓存

**流程：**
1. 加载原始FP8模型
2. 转换为float32
3. 运行生成和分析
4. 保存转换后的模型到cache_dir

#### `--dump_only`
仅转换和保存，不运行分析。

```bash
python test_minimax_m2.py --cache_dir ./cache --dump_only
```

**用途：**
- 预先准备缓存
- 不需要立即运行分析
- 节省时间（不执行生成）

**流程：**
1. 加载原始FP8模型
2. 转换为float32
3. 保存到cache_dir
4. 退出（不运行生成）

## 使用场景

### 场景1：第一次使用，想立即看结果

```bash
# 运行分析并保存缓存
python test_minimax_m2.py \
  --cache_dir ./cache \
  --dump_cache \
  --prompt "Your prompt" \
  --max_tokens 512
```

**优点：**
- 一次运行搞定
- 既有结果又有缓存

**耗时：** 约15-20分钟（转换5-10分钟 + 生成5-10分钟）

### 场景2：提前准备缓存

```bash
# 只转换和保存
python test_minimax_m2.py \
  --cache_dir ./cache \
  --dump_only
```

**优点：**
- 更快（不执行生成）
- 适合批量准备环境

**耗时：** 约5-10分钟（仅转换和保存）

### 场景3：使用已有缓存

```bash
# 直接使用缓存运行分析
python test_minimax_m2.py \
  --cache_dir ./cache \
  --prompt "Your prompt" \
  --max_tokens 1024
```

**优点：**
- 快速启动（秒级）
- 适合多次实验

**耗时：** 约5-10分钟（仅生成和分析）

### 场景4：批量实验

```bash
# 第一次：创建缓存
python test_minimax_m2.py --cache_dir ./cache --dump_only

# 后续：快速运行多个实验
for prompt_file in prompt_*.txt; do
    python test_minimax_m2.py \
        --cache_dir ./cache \
        --prompt "$prompt_file" \
        --output_dir "./results/${prompt_file%.txt}"
done
```

## 缓存目录结构

转换后的缓存目录包含：

```
model_cache_float32/
├── config.json                 # 模型配置
├── model.safetensors          # 完整模型权重（单文件）
│   或
├── model-00001-of-00003.safetensors  # 分片权重（大模型）
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json     # 分片索引
├── generation_config.json      # 生成配置
├── tokenizer.json             # Tokenizer（方便）
├── tokenizer_config.json
└── special_tokens_map.json
```

**磁盘空间：**
- MiniMax-M2 float32 缓存：约 400-500 GB
- 确保有足够的磁盘空间

## 工作原理

### 检测缓存

脚本启动时会检查：

```python
if CACHE_DIR and os.path.exists(CACHE_DIR):
    # 检查必要文件是否存在
    if os.path.exists(os.path.join(CACHE_DIR, "config.json")) and \
       os.path.exists(os.path.join(CACHE_DIR, "model.safetensors")):
        # 使用缓存
        load_from_cache = True
```

### 加载缓存

```python
model = AutoModelForCausalLM.from_pretrained(
    CACHE_DIR,  # 从缓存目录加载
    dtype=torch.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)
```

### 保存缓存

```python
model.save_pretrained(
    CACHE_DIR,
    safe_serialization=True,  # 使用safetensors格式
    max_shard_size="5GB",     # 大模型分片保存
)
```

## 最佳实践

### 1. 命名规范

使用描述性的缓存目录名：

```bash
# 好的命名
./minimax_m2_float32_cpu_cache
./cache/minimax_m2_fp32_20251030

# 不好的命名
./cache
./tmp
```

### 2. 组织结构

```bash
project/
├── model_caches/
│   ├── minimax_m2_float32/      # MiniMax-M2缓存
│   ├── deepseek_v3_float32/     # DeepSeek-V3缓存
│   └── qwen3_float32/           # Qwen3缓存
└── results/
    ├── experiment_1/
    └── experiment_2/
```

### 3. 创建一次，多次使用

```bash
# 一次性创建所有需要的缓存
python test_minimax_m2.py --cache_dir ./caches/minimax_m2 --dump_only

# 后续所有实验都使用缓存
python test_minimax_m2.py --cache_dir ./caches/minimax_m2 --prompt ...
```

### 4. 定期清理

缓存占用大量空间，不需要时及时删除：

```bash
# 删除缓存
rm -rf ./model_cache_float32

# 或者移动到备份位置
mv ./model_cache_float32 /backup/caches/
```

### 5. 共享缓存

在团队环境中，可以共享缓存：

```bash
# 服务器上创建共享缓存
python test_minimax_m2.py \
  --cache_dir /shared/caches/minimax_m2_float32 \
  --dump_only

# 团队成员使用共享缓存
python test_minimax_m2.py \
  --cache_dir /shared/caches/minimax_m2_float32 \
  --prompt "..."
```

## 常见问题

### Q1: 缓存文件损坏怎么办？

**症状：**
```
Error loading cached model: ...
```

**解决方案：**
```bash
# 删除损坏的缓存
rm -rf ./model_cache_float32

# 重新创建
python test_minimax_m2.py --cache_dir ./model_cache_float32 --dump_only
```

### Q2: 如何验证缓存是否有效？

**方法1：检查文件**
```bash
ls -lh ./model_cache_float32/
# 应该看到 config.json 和 model.safetensors
```

**方法2：尝试加载**
```bash
python test_minimax_m2.py --cache_dir ./model_cache_float32 --max_tokens 10
# 应该看到 "Loading from cache" 消息
```

### Q3: 缓存可以在不同机器间共享吗？

**可以！** safetensors格式是跨平台的。

```bash
# 在机器A上创建
python test_minimax_m2.py --cache_dir ./cache --dump_only

# 复制到机器B
rsync -av ./cache/ user@machineB:/path/to/cache/

# 在机器B上使用
python test_minimax_m2.py --cache_dir /path/to/cache --prompt "..."
```

**注意：** 
- 都必须是CPU模式
- 都必须是相同的transformers版本

### Q4: 缓存和原始模型有什么区别？

| 特性 | 原始模型 | 缓存模型 |
|------|---------|---------|
| 格式 | FP8量化 | float32 |
| 加载速度 | 中等 | 快 |
| 需要转换 | 是 | 否 |
| 磁盘空间 | 小（约100GB） | 大（约500GB） |
| 精度 | 高（FP8） | 中（FP8→FP32转换） |

### Q5: 每次修改prompt都需要重新创建缓存吗？

**不需要！** 缓存是模型权重，与prompt无关。

一次创建缓存，可以用于：
- 不同的prompts
- 不同的max_tokens
- 不同的temperature/top_p
- 不同的输出设置

### Q6: dump_only模式会生成分析报告吗？

**不会。** `--dump_only` 只执行：
1. 加载原始模型
2. 转换为float32
3. 保存缓存
4. 退出

不会执行：
- Token生成
- 专家激活分析
- 可视化生成

### Q7: 缓存占用多少空间？

**预估（MiniMax-M2）：**
- 原始FP8模型：约 100 GB
- float32缓存：约 400-500 GB
- 增加：约 300-400 GB

**计算：** FP8 (1 byte/param) → float32 (4 bytes/param) ≈ 4x

## 性能对比

### 时间对比（MiniMax-M2, CPU模式）

| 操作 | 不使用缓存 | 使用缓存 | 节省时间 |
|------|-----------|---------|---------|
| 模型加载+转换 | 5-10分钟 | 1-2分钟 | 4-8分钟 |
| 生成512 tokens | 5-10分钟 | 5-10分钟 | 0分钟 |
| **总计** | **10-20分钟** | **6-12分钟** | **4-8分钟** |

### 节省时间计算

如果你需要运行10次实验：

**不使用缓存：**
- 10次 × 15分钟（平均）= 150分钟 = **2.5小时**

**使用缓存：**
- 1次转换（10分钟）+ 10次运行 × 9分钟 = 100分钟 = **1.7小时**
- **节省：** 50分钟

## 完整示例

### 示例1：首次运行工作流

```bash
# Step 1: 创建缓存（只需一次）
echo "Step 1: Creating cache..."
python test_minimax_m2.py \
  --cache_dir ./minimax_m2_cache \
  --dump_only

echo "Cache created! Now you can run experiments quickly."

# Step 2: 运行实验
echo "Step 2: Running experiments..."
python test_minimax_m2.py \
  --cache_dir ./minimax_m2_cache \
  --prompt "Write a sorting algorithm" \
  --max_tokens 512 \
  --no_sample

# Step 3: 更多实验（都很快）
echo "Step 3: More experiments..."
python test_minimax_m2.py \
  --cache_dir ./minimax_m2_cache \
  --prompt "Explain machine learning" \
  --max_tokens 1024
```

### 示例2：批量处理脚本

```bash
#!/bin/bash
# batch_analysis.sh

CACHE_DIR="./minimax_m2_cache"

# Check if cache exists
if [ ! -d "$CACHE_DIR" ]; then
    echo "Creating cache (one-time setup)..."
    python test_minimax_m2.py --cache_dir "$CACHE_DIR" --dump_only
    echo "Cache created!"
fi

# Run multiple experiments
for prompt_file in prompts/*.txt; do
    echo "Processing: $prompt_file"
    
    output_name=$(basename "$prompt_file" .txt)
    
    python test_minimax_m2.py \
        --cache_dir "$CACHE_DIR" \
        --prompt "$prompt_file" \
        --max_tokens 1024 \
        --no_sample \
        --output_dir "./results/$output_name"
    
    echo "Completed: $output_name"
    echo "---"
done

echo "All experiments completed!"
```

## 总结

**使用缓存的好处：**
- ✅ 节省时间（每次4-8分钟）
- ✅ 快速迭代实验
- ✅ 降低CPU负载
- ✅ 一次转换，多次使用

**何时使用缓存：**
- ✅ 需要多次运行实验
- ✅ 在同一台机器上工作
- ✅ 有足够的磁盘空间
- ✅ CPU模式下运行

**何时不使用缓存：**
- ❌ 只运行一次
- ❌ 磁盘空间不足
- ❌ 使用GPU模式（不需要转换）
- ❌ 模型会频繁更新

**推荐工作流：**
1. 首次：`--dump_only` 创建缓存
2. 后续：`--cache_dir` 使用缓存
3. 完成：删除缓存释放空间

开始使用缓存，让你的实验更高效！🚀

