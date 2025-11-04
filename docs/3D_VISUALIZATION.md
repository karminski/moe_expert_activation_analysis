# 3D专家激活可视化指南

## 概述

新增的3D可视化功能使用Plotly创建交互式3D表面图，展示专家激活概率在多个维度上的分布：
- **X轴**: 专家索引
- **Y轴**: Token位置（时间序列）
- **Z轴**: 激活概率（每层有不同的偏移量用于堆叠显示）

## 特点

### 🎨 视觉特性
- **多层堆叠**: 多个MoE层垂直堆叠，便于对比
- **热力颜色**: 从冷色（低激活）到热色（高激活）的渐变
- **透明度**: 0.85的透明度，可以看到下层的模式
- **等轴测视图**: 类似现代数据可视化的风格

### 🖱️ 交互功能
- **旋转**: 点击并拖拽可以从任意角度查看
- **缩放**: 滚轮或双指缩放
- **平移**: Shift + 拖拽平移视图
- **悬停**: 鼠标悬停显示详细数值
  - 专家索引
  - Token位置
  - 激活概率
  - 层索引
- **导出**: 工具栏中的相机图标可以保存为PNG图片

## 如何使用

### 查看3D可视化

运行分析后，在输出目录中找到 `expert_activation_3d.html`：

```bash
# 在浏览器中打开
# Windows
start ./minimax_m2_results_*/expert_activation_3d.html

# Linux/Mac
xdg-open ./minimax_m2_results_*/expert_activation_3d.html

# 或直接用浏览器打开
# Chrome, Firefox, Safari, Edge 都支持
```

### 性能优化参数

3D可视化默认限制了数据量以保证流畅性：

```python
visualizer.plot_expert_activation_3d(
    analyzer,
    save_path="expert_activation_3d.html",
    max_tokens=50,   # 最多显示50个token
    max_layers=10,   # 最多显示10层
    max_experts=64,  # 最多显示64个专家
)
```

**调整建议：**
- 如果模型较小或想看更多细节，可以增加这些值
- 如果浏览器卡顿，可以减少这些值
- MiniMax-M2 (62层, 256专家) 默认值已优化

### 自定义3D可视化

在你的脚本中：

```python
from visualizer import MoEVisualizer
from moe_analyzer import MoEAnalyzer

# ... (加载模型和收集数据)

visualizer = MoEVisualizer()

# 创建自定义3D可视化
visualizer.plot_expert_activation_3d(
    analyzer,
    save_path="./my_3d_viz.html",
    max_tokens=100,    # 显示更多tokens
    max_layers=20,     # 显示更多layers  
    max_experts=128,   # 显示更多experts
)
```

## 解读3D可视化

### 寻找什么

#### 1. **峰值模式**
- **高尖峰**: 特定专家在特定token上强烈激活
- **平坦区域**: 激活分布均匀
- **山脊**: 某些专家持续被激活

#### 2. **层间比较**
- **相似形状**: 不同层的激活模式相似（可能表示周期性）
- **演化趋势**: 从下层到上层激活模式如何变化
- **突变**: 某一层的模式突然改变

#### 3. **时间序列模式**
- **沿Y轴的波动**: Token序列中的激活变化
- **周期性**: 每隔几个token重复的模式
- **异常值**: 特定token的异常激活

### 示例解读

#### 模式A: 专家专业化
```
层1: 平坦，激活分散
层2: 平坦，激活分散
层3: 出现明显峰值，特定专家开始专业化
层4: 峰值更明显，专家分工清晰
```
**解释**: 模型在深层开始进行专家分工

#### 模式B: 周期性激活
```
层1和层13: 激活模式相似（Δ=12）
层2和层14: 激活模式相似（Δ=12）
```
**解释**: 存在12层的周期性模式

#### 模式C: Token依赖
```
Token 0-10: 低激活（可能是padding）
Token 11-50: 高且变化的激活（实际内容）
```
**解释**: 模型对不同位置的token反应不同

## 与2D热力图的对比

| 特性 | 2D热力图 | 3D可视化 |
|------|---------|---------|
| **维度** | 层 × 专家 | 层 × Token × 专家 |
| **时间信息** | ❌ 平均值 | ✅ 保留时间序列 |
| **交互性** | ✅ 缩放、悬停 | ✅✅ 旋转、缩放、悬停 |
| **数据密度** | 高（所有数据） | 中（采样显示） |
| **适合场景** | 整体趋势 | 详细分析、演示 |
| **加载速度** | 快 | 中等 |

**建议：**
- 先看2D热力图了解整体
- 再看3D可视化深入细节
- 两者结合使用效果最佳

## 技术细节

### 数据采样
为了性能，3D可视化会对数据进行采样：
- **层采样**: `layer_step = max(1, num_layers // max_layers)`
- **Token截断**: 只取前 `max_tokens` 个
- **专家截断**: 只取前 `max_experts` 个

### Z轴偏移
每层增加 0.3 的Z轴偏移：
```python
z_offset = layer_idx * 0.3
z_data = activation_probs + z_offset
```

这样不同层可以堆叠显示而不重叠。

### 颜色映射
使用自定义蓝-粉渐变方案（优化可读性）：
- 0.0 (低激活): #19448e 深蓝色
- 0.25: #4a6fa5 中蓝色
- 0.5 (中等激活): #7d8db8 浅蓝紫色
- 0.75: #c49fbb 浅紫粉色
- 1.0 (高激活): #f4b3c2 粉红色

这个配色方案相比默认的"Hot"颜色：
- ✅ 避免了白色导致的视觉混淆
- ✅ 提供清晰的低-高激活对比
- ✅ 在深色和浅色背景下都易读
- ✅ 色彩过渡平滑自然

### 视角设置
默认使用等轴测视图：
```python
camera = dict(
    eye=dict(x=1.8, y=-1.8, z=1.5),  # 观察点
    center=dict(x=0, y=0, z=0),       # 焦点
    up=dict(x=0, y=0, z=1),           # 上方向
)
```

## 常见问题

### Q1: 浏览器打开很慢？
**A**: 这是正常的，3D可视化包含大量数据。
- 尝试减少 `max_tokens`, `max_layers`, `max_experts` 参数
- 使用现代浏览器（Chrome/Edge推荐）
- 关闭其他标签页释放内存

### Q2: 看不清细节？
**A**: 使用交互功能：
- 旋转到不同角度
- 放大感兴趣的区域
- 悬停查看具体数值

### Q3: 如何保存视图？
**A**: 
1. 调整到满意的角度
2. 点击右上角的相机图标
3. 选择 "Download plot as png"

### Q4: 能否更改颜色方案？
**A**: 可以！编辑 `visualizer.py` 中的 `plot_expert_activation_3d` 方法。

当前使用的是自定义蓝-粉渐变（优化可读性）：
```python
custom_colorscale = [
    [0.0, '#19448e'],   # 深蓝色
    [0.25, '#4a6fa5'],  # 中蓝色
    [0.5, '#7d8db8'],   # 浅蓝紫色
    [0.75, '#c49fbb'],  # 浅紫粉色
    [1.0, '#f4b3c2']    # 粉红色
]
```

也可以改为Plotly内置方案：
```python
colorscale='Viridis'  # 或 'Plasma', 'Blues', 'Reds', 'Greens' 等
```

### Q5: 如何对比两次运行的结果？
**A**: 
1. 在浏览器中打开两个3D可视化文件
2. 并排窗口查看
3. 或使用屏幕截图工具记录不同角度

## 高级用法

### 创建动画GIF

使用 Plotly 的帧功能（需要修改代码）：

```python
# 创建旋转动画
frames = []
for angle in range(0, 360, 10):
    eye_x = 1.8 * np.cos(np.radians(angle))
    eye_y = 1.8 * np.sin(np.radians(angle))
    frames.append(go.Frame(
        layout=dict(
            scene_camera=dict(
                eye=dict(x=eye_x, y=eye_y, z=1.5)
            )
        )
    ))

fig.frames = frames
```

### 导出到其他格式

使用 `kaleido` 库（需安装）：

```bash
pip install kaleido
```

```python
import plotly.graph_objects as go

fig = go.Figure(...)
fig.write_image("output.png", width=1920, height=1080)
fig.write_image("output.svg")  # 矢量图
```

### 集成到Jupyter Notebook

```python
from visualizer import MoEVisualizer

visualizer = MoEVisualizer()
visualizer.plot_expert_activation_3d(analyzer, save_path="temp.html")

# 在notebook中直接显示
from IPython.display import IFrame
IFrame(src="temp.html", width=1400, height=1000)
```

## 示例展示

### 场景1：发现周期性模式
```
观察: 层0, 12, 24的3D表面形状相似
结论: 存在Δ=12的周期性
用途: 验证层间upcycling假设
```

### 场景2：专家专业化分析
```
观察: 
- 浅层（0-10）: 激活分散
- 深层（50-61）: 特定专家高激活
结论: 模型在深层进行任务专业化
用途: 理解模型如何处理不同类型的输入
```

### 场景3：异常检测
```
观察: 某一token位置所有层都显示异常高激活
可能原因: 
- 特殊token（如EOS）
- 输入中的异常值
- 模型的注意力焦点
```

## 相关文档

- [README.md](README.md) - 完整项目文档
- [USAGE.md](USAGE.md) - 使用指南
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - 故障排除

## 参考

3D可视化的灵感来源：
- [Twitter讨论](https://x.com/kilian_maciej/status/1982612297874026731) 关于Qwen3的深度wise层upcycling
- 现代数据可视化最佳实践
- Plotly 3D表面图文档

---

享受探索MoE模型的3D世界！🚀
