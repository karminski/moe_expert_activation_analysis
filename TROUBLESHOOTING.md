# 故障排除指南 (Troubleshooting Guide)

## 常见问题和解决方案

### 1. 生成的token数量太少

**症状：**
```
✅ Generation completed!
📌 Total tokens: 11
📌 Generated tokens: 2
```
只生成了2个token，远少于预期。

**原因：**
- 模型过早生成了EOS（结束）token
- 在CPU上运行FP8转float32模型可能导致输出质量下降
- 生成参数（temperature, top_p）可能不适合

**解决方案：**

#### 方案1：使用贪婪解码（推荐）
```bash
python test_minimax_m2.py \
  --prompt "Write a Python function to calculate fibonacci numbers" \
  --max_tokens 1024 \
  --no_sample
```
贪婪解码通常更稳定，特别是在CPU模式下。

#### 方案2：降低temperature
```bash
python test_minimax_m2.py \
  --prompt "Write a Python function to calculate fibonacci numbers" \
  --max_tokens 1024 \
  --temperature 0.3 \
  --top_p 0.95
```
较低的temperature使输出更确定性。

#### 方案3：更改prompt
```bash
# 更具体的prompt
python test_minimax_m2.py \
  --prompt "Below is a complete Python implementation:\n\ndef fibonacci(n):"
```

#### 方案4：使用GPU（如果可用）
在 `test_minimax_m2.py` 中设置：
```python
USE_CPU_MODE = False
```

**检查生成参数：**
脚本现在会显示详细的生成参数：
```
⚙️  Generation parameters:
   - max_new_tokens: 1024
   - do_sample: True
   - temperature: 0.7
   - top_p: 0.9
   - pad_token_id: ...
```

### 2. 生成的文本包含乱码

**症状：**
```
Generated Text: Write a Python function to calculate fibonacci numbers群的
```

**原因：**
- FP8量化模型在CPU上转换为float32后质量下降
- Tokenizer解码问题
- 模型输出的logits分布异常

**解决方案：**

#### 方案1：使用贪婪解码
```bash
python test_minimax_m2.py --no_sample --max_tokens 512
```

#### 方案2：检查tokenizer配置
脚本会显示：
```
🔍 Tokenizer info:
   - eos_token_id: ...
   - pad_token_id: ...
   - bos_token_id: ...
```

#### 方案3：使用GPU
CPU模式下FP8→float32转换可能导致精度损失。如果有GPU，尽量使用GPU。

### 3. 内存不足 (OOM)

**症状：**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:...]
```

**解决方案：**

#### 降低生成长度
```bash
python test_minimax_m2.py --max_tokens 256
```

#### 禁用专家权重相似度分析
```bash
python test_minimax_m2.py  # 默认就是禁用的
```

#### 减少并行线程数
```bash
python test_minimax_m2.py --enable_expert_similarity --n_jobs 16
```

### 4. 生成速度太慢

**症状：**
在CPU上生成100个token需要5-10分钟以上。

**这是正常的！** MiniMax-M2是一个超大型模型（62层，256个专家）。

**加速方法：**

#### 使用GPU
最有效的方法。在脚本中设置：
```python
USE_CPU_MODE = False
```

#### 减少生成长度
```bash
python test_minimax_m2.py --max_tokens 256
```

#### 使用贪婪解码（略快）
```bash
python test_minimax_m2.py --no_sample
```

### 5. 专家权重相似度计算太慢

**症状：**
卡在：
```
[5/6] Computing expert weight similarity (delta=12)...
```
超过10分钟。

**解决方案：**

#### 方案1：启用并行计算（推荐）
```bash
python test_minimax_m2.py \
  --enable_expert_similarity \
  --n_jobs 100  # 使用100个核心
```

#### 方案2：禁用该功能（默认）
```bash
python test_minimax_m2.py  # 不加--enable_expert_similarity
```

#### 方案3：只在需要时运行
专家权重相似度分析很有价值但耗时。建议：
1. 先运行快速分析（不启用）
2. 查看其他结果
3. 如果需要完整报告，再运行一次启用该功能

### 6. Import错误

**症状：**
```
ModuleNotFoundError: No module named 'torch'
```

**解决方案：**
```bash
# 安装依赖
pip install -r requirements.txt

# 或者单独安装
pip install torch transformers numpy scipy plotly
```

### 7. 文件编码错误（Windows）

**症状：**
```
UnicodeEncodeError: 'gbk' codec can't encode character
```

**解决方案：**

#### PowerShell
```powershell
$env:PYTHONIOENCODING="utf-8"
python test_minimax_m2.py
```

#### CMD
```cmd
set PYTHONIOENCODING=utf-8
python test_minimax_m2.py
```

#### Git Bash / WSL
```bash
export PYTHONIOENCODING=utf-8
python test_minimax_m2.py
```

### 8. 帮助选项不显示

**症状：**
```bash
python test_minimax_m2.py --help
# 显示错误
```

**原因：**
缺少依赖（torch等），脚本在import时就失败了。

**解决方案：**
先安装依赖：
```bash
pip install -r requirements.txt
```

### 9. Prompt文件找不到

**症状：**
```
[*] Loading prompt from file: my_prompt.txt
[!] Error reading file: [Errno 2] No such file or directory
```

**解决方案：**

#### 使用绝对路径
```bash
python test_minimax_m2.py --prompt "C:/path/to/my_prompt.txt"
```

#### 使用相对路径
```bash
python test_minimax_m2.py --prompt "./my_prompt.txt"
```

#### 检查文件是否存在
```bash
# Windows PowerShell
Test-Path my_prompt.txt

# Linux/Mac/Git Bash
ls my_prompt.txt
```

### 10. 生成参数不生效

**症状：**
指定了 `--max_tokens 1024`，但还是只生成了512个token。

**检查：**
- 查看脚本输出的配置信息
- 确认使用的是 `max_new_tokens` 而不是 `max_length`
- 检查是否遇到EOS token（参见问题1）

**验证：**
```bash
python test_minimax_m2.py --max_tokens 100
# 应该显示：
# 📊 Max New Tokens: 100
```

## 调试技巧

### 1. 查看详细输出

脚本已经包含了详细的调试信息：
```
🔍 Tokenizer info:
   - eos_token_id: ...
   - pad_token_id: ...

⚙️  Generation parameters:
   - max_new_tokens: ...
   - do_sample: ...
```

### 2. 测试小规模配置

在调试时使用最小配置：
```bash
python test_minimax_m2.py \
  --max_tokens 64 \
  --no_sample \
  --disable_structured_output
```

### 3. 检查模型加载

观察模型加载输出：
```
✅ Model loaded successfully
🔄 Converting all weights to float32...
✅ All weights converted to float32
```

如果卡在加载阶段，可能是：
- 内存不足
- 模型文件损坏
- 路径错误

### 4. 逐步排查

1. **第一步：能否运行？**
   ```bash
   python test_minimax_m2.py --max_tokens 10
   ```

2. **第二步：能否生成？**
   检查输出的token数量

3. **第三步：生成质量如何？**
   查看生成文本是否合理

4. **第四步：能否完整分析？**
   运行完整配置

## 性能基准

### 参考时间（CPU模式，100核心服务器）

| 任务 | max_tokens | 时间 |
|------|-----------|------|
| 快速测试 | 128 | 2-3分钟 |
| 标准分析 | 512 | 5-10分钟 |
| 长文本分析 | 1024 | 10-20分钟 |
| +专家权重（串行） | 512 | 15-30分钟 |
| +专家权重（并行100核） | 512 | 10-15分钟 |

### GPU模式（参考）

| 任务 | max_tokens | 时间（GPU） |
|------|-----------|------------|
| 标准分析 | 512 | 30秒-2分钟 |
| 长文本分析 | 1024 | 1-3分钟 |
| +专家权重 | 512 | 2-5分钟 |

## 仍然有问题？

### 收集信息

运行以下命令收集调试信息：
```bash
python test_minimax_m2.py --max_tokens 10 > debug_output.txt 2>&1
```

### 检查事项

1. **Python版本**
   ```bash
   python --version  # 推荐 3.8+
   ```

2. **依赖版本**
   ```bash
   pip list | grep -E "torch|transformers|numpy"
   ```

3. **系统资源**
   ```bash
   # Linux
   free -h
   nproc
   
   # Windows PowerShell
   Get-ComputerInfo | Select-Object CsProcessors, OsTotalVisibleMemorySize
   ```

4. **模型文件**
   ```bash
   ls -lh /hc550x10rz2-01/llms/MiniMax/MiniMax-M2/
   ```

### 联系支持

提供以下信息：
- 完整的错误消息
- 使用的命令
- Python和依赖版本
- 系统信息（CPU核心数，内存）
- `debug_output.txt` 文件

## 已知限制

1. **CPU模式下的FP8模型**
   - 质量可能下降
   - 生成速度慢
   - 建议使用GPU

2. **超长文本生成**
   - 2048+ tokens在CPU上非常慢
   - 考虑分批生成

3. **专家权重相似度**
   - 计算密集，即使并行也需要时间
   - 对于完整62层×256专家，建议至少64核心并行

4. **Windows字符集问题**
   - emoji可能显示为乱码
   - 设置 PYTHONIOENCODING=utf-8

## 最佳实践

1. **开发/调试**：使用 `--max_tokens 128 --no_sample`
2. **标准分析**：使用 `--max_tokens 512`
3. **完整分析**：使用 `--max_tokens 1024 --enable_expert_similarity --n_jobs 64`
4. **生产使用**：使用GPU模式
5. **批量实验**：禁用专家权重相似度，加快处理

---

如果本文档没有解决你的问题，请提issue并提供详细的错误信息！

