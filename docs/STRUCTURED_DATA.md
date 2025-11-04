# 结构化数据输出说明

## 概述

MoE分析工具现在支持导出结构化数据（JSON/JSONL/Pickle格式），方便程序化处理和AI分析。

## 配置参数

在 `test_minimax_m2.py` 中配置：

```python
# 启用结构化数据输出
ENABLE_STRUCTURED_OUTPUT = True  # 设置为True以生成结构化数据

# 选择输出格式
STRUCTURED_OUTPUT_FORMAT = "json"  # 可选：json, jsonl, pickle
```

## 输出文件

启用后会生成以下文件：

### 1. `analysis_data.json` - 完整分析数据
包含所有分析结果的详细数据，适合深度分析。

### 2. `analysis_summary.json` - 分析摘要
紧凑的摘要数据，包含关键发现，适合快速查看。

## 数据结构

### analysis_data.json 结构

```json
{
  "metadata": {
    "model_info": {
      "num_moe_layers": 62,
      "layer_indices": [0, 1, 2, ...],
      "total_tokens_analyzed": 167,
      "num_experts_per_layer": {...},
      "top_k_per_layer": {...}
    },
    "export_format": "json",
    "include_raw_data": false
  },
  "expert_activation": {
    "matrix": [[...], [...], ...],  // [layers, experts]
    "layer_indices": [0, 1, 2, ...],
    "per_layer_stats": {
      "0": {
        "avg_activation": [...],
        "var_activation": [...],
        "num_tokens": 167,
        "num_experts": 256
      },
      ...
    }
  },
  "layer_correlation": {
    "correlation_matrix": [[...], ...],  // [layers, deltas]
    "deltas": [1, 2, 3, ...],
    "layer_indices": [0, 1, 2, ...],
    "layer_pairs": [
      {
        "layer_i": 0,
        "layer_j": 12,
        "delta": 12,
        "correlation": 0.85
      },
      ...
    ]
  },
  "periodic_patterns": {
    "12": {
      "mean_correlation": 0.85,
      "std_correlation": 0.12,
      "num_pairs": 50,
      "all_correlations": [0.82, 0.88, ...]
    },
    "24": {...},
    "36": {...}
  },
  "router_similarity": {
    "cosine_similarity_matrix": [[...], ...],
    "column_norm_correlation": [[...], ...],
    "router_layers": [0, 1, 2, ...]
  }
}
```

### analysis_summary.json 结构

```json
{
  "model": {
    "num_moe_layers": 62,
    "layer_indices": [0, 1, 2, ...],
    "total_tokens_analyzed": 167
  },
  "key_findings": {
    "num_moe_layers": 62,
    "periodic_patterns": {
      "12": {
        "mean_correlation": 0.85,
        "strength": "strong"  // strong/moderate/weak
      },
      "24": {
        "mean_correlation": 0.42,
        "strength": "moderate"
      }
    }
  }
}
```

## 使用示例

### Python 读取和分析

```python
import json
import numpy as np

# 读取完整分析数据
with open('analysis_data.json', 'r') as f:
    data = json.load(f)

# 获取专家激活矩阵
activation_matrix = np.array(data['expert_activation']['matrix'])
print(f"Shape: {activation_matrix.shape}")  # (62, 256)

# 分析周期性模式
for delta, pattern in data['periodic_patterns'].items():
    print(f"Delta={delta}: correlation={pattern['mean_correlation']:.3f}")

# 查看层间相关性
for pair in data['layer_correlation']['layer_pairs']:
    if pair['delta'] == 12 and pair['correlation'] > 0.8:
        print(f"Strong correlation: Layer {pair['layer_i']} -> {pair['layer_j']}")

# 读取摘要
with open('analysis_summary.json', 'r') as f:
    summary = json.load(f)

print(f"Total MoE layers: {summary['model']['num_moe_layers']}")
for delta, info in summary['key_findings']['periodic_patterns'].items():
    print(f"Δ={delta}: {info['strength']} pattern (r={info['mean_correlation']:.3f})")
```

### 使用 pandas 分析

```python
import pandas as pd
import json

# 读取数据
with open('analysis_data.json', 'r') as f:
    data = json.load(f)

# 转换层间相关性为DataFrame
df = pd.DataFrame(data['layer_correlation']['layer_pairs'])
print(df.head())

# 筛选高相关性的层对
high_corr = df[df['correlation'] > 0.7]
print(f"Found {len(high_corr)} highly correlated layer pairs")

# 分析特定delta的模式
delta_12 = df[df['delta'] == 12]
print(f"Delta=12 mean correlation: {delta_12['correlation'].mean():.3f}")

# 导出为CSV供其他工具使用
delta_12.to_csv('layer_correlation_delta12.csv', index=False)
```

### JavaScript/Node.js 读取

```javascript
const fs = require('fs');

// 读取分析数据
const data = JSON.parse(fs.readFileSync('analysis_data.json', 'utf8'));

// 获取专家激活统计
const expertActivation = data.expert_activation;
console.log(`Analyzed ${expertActivation.layer_indices.length} layers`);

// 查找周期性模式
Object.entries(data.periodic_patterns).forEach(([delta, pattern]) => {
    console.log(`Δ=${delta}: r=${pattern.mean_correlation.toFixed(3)}`);
});
```

### 使用 AI 分析（提示词示例）

将 `analysis_summary.json` 提供给AI：

```
请分析以下MoE模型的专家激活数据：

[粘贴 analysis_summary.json 内容]

问题：
1. 是否存在明显的周期性模式？
2. 哪个delta值显示最强的相关性？
3. 这可能表明什么样的训练或架构特点？
```

## 输出格式对比

| 格式 | 优点 | 缺点 | 使用场景 |
|------|------|------|----------|
| **JSON** | 人类可读、通用性强 | 文件较大 | 分享、AI分析、Web应用 |
| **JSONL** | 流式读取、增量处理 | 不够直观 | 大数据处理、日志分析 |
| **Pickle** | 保留Python对象、最小 | 仅限Python | Python深度分析 |

## 高级选项

### 包含原始数据

如果需要完整的路由概率矩阵（会显著增加文件大小）：

```python
# 在 visualizer.export_structured_data() 调用中
visualizer.export_structured_data(
    analyzer,
    output_dir=OUTPUT_DIR,
    format="json",
    include_raw_data=True  # 包含每个token的路由概率
)
```

这会在 `per_layer_stats` 中添加：
- `routing_probs`: 每个token的路由概率 [tokens, experts]
- `selected_experts`: 每个token选择的专家索引 [tokens, top_k]

⚠️ 警告：对于长序列，这会生成非常大的JSON文件（数百MB到GB）。

## 数据说明

### 相关性强度判定

- **Strong (强)**: correlation > 0.7
- **Moderate (中等)**: 0.4 < correlation ≤ 0.7
- **Weak (弱)**: correlation ≤ 0.4

### NaN 值处理

相关性矩阵中的 `null` 值表示：
- 该层对不存在（超出层数范围）
- 或者计算失败（数据不足等）

### 层索引

- 所有层索引从 0 开始
- `layer_indices` 列表显示哪些层包含MoE

## 疑难解答

### Q: JSON文件太大怎么办？
A: 
1. 设置 `include_raw_data=False`（默认）
2. 使用 `jsonl` 格式按行读取
3. 使用 `pickle` 格式（Python专用）

### Q: 如何只获取特定层的数据？
A: 在Python中过滤：
```python
layers_of_interest = [0, 12, 24, 36]
filtered_stats = {
    k: v for k, v in data['expert_activation']['per_layer_stats'].items()
    if int(k) in layers_of_interest
}
```

### Q: 如何合并多次运行的结果？
A: 
```python
import json

results = []
for i in range(5):
    with open(f'run_{i}/analysis_summary.json') as f:
        results.append(json.load(f))

# 计算平均相关性
avg_corr_12 = sum(r['key_findings']['periodic_patterns']['12']['mean_correlation'] 
                  for r in results) / len(results)
```

## 示例：检测层重复模式

```python
import json
import numpy as np

with open('analysis_data.json') as f:
    data = json.load(f)

# 获取所有delta=12的相关性
pairs = data['layer_correlation']['layer_pairs']
delta_12_pairs = [p for p in pairs if p['delta'] == 12]

# 找出高度相关的层对
high_corr_pairs = [p for p in delta_12_pairs if p['correlation'] > 0.8]

print(f"Found {len(high_corr_pairs)} layer pairs with high similarity (Δ=12)")
print("\nPotential layer stacking/upcycling detected:")
for p in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True):
    print(f"  Layer {p['layer_i']:2d} → {p['layer_j']:2d}: r={p['correlation']:.3f}")
```

## 相关文件

- `test_minimax_m2.py` - 主测试脚本（包含配置参数）
- `visualizer.py` - 包含 `export_structured_data()` 方法
- `moe_analyzer.py` - 数据分析核心逻辑

