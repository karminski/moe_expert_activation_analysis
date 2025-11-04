# CPU运行指南

## 问题说明

如果你在运行时遇到以下错误：

```
FP8 quantized models is only supported on GPUs with compute capability >= 8.9
```

这是因为模型可能包含FP8量化配置，但你的GPU不支持或你想使用CPU运行。

## 解决方案

### 方案1：使用修改后的test_minimax_m2.py（推荐）

脚本已经配置为在CPU上运行。关键修改点：

1. **在文件开头设置环境变量**（第11行）：
```python
USE_CPU_MODE = True  # 设置为False以使用GPU

if USE_CPU_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 必须在导入torch之前
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
```

2. **使用正确的参数加载模型**（第65-71行）：
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,  # CPU使用float32，不用torch_dtype
    device_map={"": "cpu"},  # 强制使用CPU
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    quantization_config=None,  # 禁用量化
)
```

**直接运行**：
```bash
python test_minimax_m2.py
```

### 方案2：使用Shell脚本（Linux/Mac）

```bash
./run_minimax_m2_cpu.sh
```

或手动设置环境变量：
```bash
export CUDA_VISIBLE_DEVICES=""
python test_minimax_m2.py
```

### 方案3：如果模型配置文件包含FP8设置

如果上述方法仍然报错，可能需要修改模型的配置文件。

1. **查看模型配置**：
```bash
cat /hc550x10rz2-01/llms/MiniMax/MiniMax-M2/config.json
```

2. **检查是否有量化配置**，如果有类似以下内容：
```json
{
  "quantization_config": {
    "quant_method": "fp8"
  }
}
```

3. **临时修改配置**（创建本地副本）：
```python
from transformers import AutoConfig, AutoModelForCausalLM
import torch

# 加载配置
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 移除量化配置
config.quantization_config = None

# 使用修改后的配置加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    dtype=torch.float32,
    device_map={"": "cpu"},
    trust_remote_code=True,
)
```

## 验证CPU模式

运行脚本时，你应该看到：

```
⚠️  CPU Mode Enabled (CUDA disabled via environment variable)
    Note: CPU inference will be significantly slower than GPU.
    For large models, this may take considerable time.
    CUDA available: False
    CUDA device count: 0
```

如果 `CUDA available: False`，说明成功禁用了CUDA。

## 性能提示

### CPU运行速度优化

1. **减少生成长度**：
```python
MAX_LENGTH = 256  # 从512减少到256
```

2. **使用更少的周期分析间隔**：
```python
PERIODIC_INTERVALS = [12]  # 只检查Δ=12
```

3. **使用PyTorch优化**：
```python
torch.set_num_threads(os.cpu_count())  # 使用所有CPU核心
```

### 预期时间

对于MiniMax-M2在CPU上的运行时间参考：
- 模型加载：5-10分钟
- 生成512 tokens：30-60分钟（取决于CPU性能）
- 分析和可视化：1-2分钟

## 切换到GPU模式

如果你有支持的GPU（compute capability >= 8.9），修改第11行：

```python
USE_CPU_MODE = False  # 改为False
```

然后运行：
```bash
python test_minimax_m2.py
```

## 常见问题

### Q: 为什么必须在导入torch之前设置环境变量？
A: PyTorch在导入时会初始化CUDA，一旦初始化完成就无法更改。因此必须在导入前通过环境变量禁用。

### Q: 可以使用float16加速CPU运行吗？
A: 不建议。大多数CPU不支持float16，强制使用会导致性能下降或错误。建议使用float32。

### Q: 内存不足怎么办？
A: 
1. 关闭其他程序
2. 使用 `low_cpu_mem_usage=True`（已启用）
3. 考虑使用量化版本（但需要特殊处理）
4. 减少生成长度

### Q: 可以中断后继续吗？
A: 当前版本不支持断点续传。如果中断，需要重新运行。

## 技术细节

### 为什么需要禁用FP8量化？

FP8（8位浮点）是一种需要特殊硬件支持的量化格式：
- 需要GPU compute capability >= 8.9（如H100, RTX 4090）
- CPU不支持FP8计算
- 必须转换为float32或float16才能在CPU上运行

### device_map参数说明

```python
device_map={"": "cpu"}  # 强制所有层到CPU
device_map="auto"        # 自动分配（可能选择GPU）
```

## 联系与支持

如果遇到其他问题，请检查：
1. transformers版本：`pip show transformers`
2. torch版本：`python -c "import torch; print(torch.__version__)"`
3. 模型文件是否完整：`ls -lh /hc550x10rz2-01/llms/MiniMax/MiniMax-M2/`

