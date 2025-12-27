# mIOU 计算公式说明

## IoU（单个部分类别）的计算公式

在代码中（`train_partseg.py` 第 244-245 行）：

```python
part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
    np.sum((segl == l) | (segp == l)))
```

### 公式

对于每个部分类别 `l`：

```
IoU(l) = |预测为l 且 真实为l| / |预测为l 或 真实为l|
```

用集合表示：
```
IoU(l) = |Pred ∩ True| / |Pred ∪ True|
```

其中：
- `|Pred ∩ True|`：预测为 l 且真实为 l 的点数（**交集**）
- `|Pred ∪ True|`：预测为 l 或真实为 l 的点数（**并集**）

### 特殊情况

如果真实和预测都没有该部分：
```python
if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
    IoU = 1.0  # 避免分母为0
```

## mIOU（多个部分类别的平均）的计算公式

对每个物体，计算其所有部分类别的 IoU 的平均值：

```
mIOU(物体) = (IoU(part1) + IoU(part2) + ... + IoU(partN)) / N
```

代码实现（第 246 行）：
```python
shape_ious[cat].append(np.mean(part_ious))
```

## 具体示例

假设一个 Airplane 物体有 4 个部分类别：`[0, 1, 2, 3]`

### 示例数据
- 总点数：1000
- **部分 0（机身）**：真实 300 点，预测 280 点，正确 270 点
- **部分 1（机翼）**：真实 200 点，预测 220 点，正确 190 点
- **部分 2（尾翼）**：真实 100 点，预测 0 点，正确 0 点
- **部分 3（其他）**：真实 400 点，预测 500 点，正确 400 点

### 计算每个部分的 IoU

**部分 0（机身）：**
```
交集 = 270点（预测正确）
并集 = 300（真实） + 280（预测） - 270（交集） = 310点
IoU(0) = 270 / 310 = 0.871
```

**部分 1（机翼）：**
```
交集 = 190点
并集 = 200 + 220 - 190 = 230点
IoU(1) = 190 / 230 = 0.826
```

**部分 2（尾翼）：**
```
交集 = 0点
并集 = 100 + 0 - 0 = 100点
IoU(2) = 0 / 100 = 0.0
```

**部分 3（其他）：**
```
交集 = 400点
并集 = 400 + 500 - 400 = 500点
IoU(3) = 400 / 500 = 0.8
```

### 计算该物体的 mIOU

```
mIOU = (0.871 + 0.826 + 0.0 + 0.8) / 4 = 0.624
```

## 可视化理解

```
IoU = 交集 / 并集

真实标签:  [●●●○○○○○○]  (3个点属于该部分)
预测标签:  [●●○○○○○○○○]  (2个点预测为该部分)
交集:      [●●○○○○○○○○]  (2个点正确预测)
并集:      [●●●○○○○○○○]  (3个点，并集大小)

IoU = 2 / 3 = 0.667
```

## 三个评价指标的区别

### 1. Test Accuracy（准确率）

```python
accuracy = total_correct / float(total_seen)
```

- **计算方式**：正确预测的点数 / 总点数
- **含义**：逐点分类准确率，衡量有多少点被正确分类

### 2. Class avg mIOU（类别平均 mIOU）

```python
# 步骤1: 计算每个物体的mIOU
shape_ious[cat].append(np.mean(part_ious))

# 步骤2: 计算每个类别的平均mIOU
shape_ious[cat] = np.mean(shape_ious[cat])

# 步骤3: 计算所有类别的平均
mean_shape_ious = np.mean(list(shape_ious.values()))
```

- **计算方式**：
  1. 对每个物体，计算其所有部分类别的 IoU 平均值（得到该物体的 mIOU）
  2. 对每个物体类别（如 Airplane、Chair），计算该类所有物体的 mIOU 平均值
  3. 对所有 16 个物体类别的平均 mIOU 再取平均
- **含义**：各类别分割质量的平均，平衡不同类别的影响

### 3. Instance avg IOU（实例平均 IoU）

```python
all_shape_ious = []
for cat in shape_ious.keys():
    for iou in shape_ious[cat]:
        all_shape_ious.append(iou)
inctance_avg_iou = np.mean(all_shape_ious)
```

- **计算方式**：收集所有物体的 mIOU，直接取平均
- **含义**：所有物体分割质量的平均，不区分类别

## 指标对比表

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **Test Accuracy** | 正确点数 / 总点数 | 逐点分类准确率 |
| **Class avg mIOU** | 先按类别平均，再跨类别平均 | 各类别分割质量的平均（平衡类别） |
| **Instance avg IOU** | 所有物体的 mIOU 直接平均 | 所有物体分割质量的平均（整体评估） |

## 示例说明

假设测试集有：
- 10 个 Airplane（平均 mIOU = 0.85）
- 10 个 Chair（平均 mIOU = 0.75）

计算：
- **Class avg mIOU** = (0.85 + 0.75) / 2 = 0.80（先按类别平均，再跨类别平均）
- **Instance avg IOU** = 所有 20 个物体的 mIOU 的平均值（直接平均所有物体）

## 总结

1. **单个部分类别的 IoU**：
   ```
   IoU = (预测正确且真实存在) / (预测存在或真实存在)
   ```

2. **物体的 mIOU**：
   ```
   mIOU = 所有部分类别IoU的平均值
   ```

3. **类别平均 mIOU**：
   ```
   Class avg mIOU = 所有类别的平均mIOU的平均值
   ```

4. **实例平均 IoU**：
   ```
   Instance avg IOU = 所有物体的mIOU的平均值
   ```

IoU 衡量预测与真实的重叠程度，范围 [0, 1]，1 表示完全匹配。

