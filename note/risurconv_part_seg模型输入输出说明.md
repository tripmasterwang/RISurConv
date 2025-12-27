# RISurConv 部件分割模型输入输出说明

## 一、模型初始化 (`__init__`)

### 初始化参数

```python
def __init__(self, num_class, normal_channel=True)
```

**参数说明：**
- `num_class` (int): 分割类别数，通常为 **50**（ShapeNet 有 50 个部件类别）
- `normal_channel` (bool): 是否使用法向量通道，默认 `True`

### 模型结构

模型包含以下主要组件：

1. **编码器（Encoder）- Set Abstraction 层：**
   - `sa0`: 2048 → 512 点，输出 64 维特征
   - `sa1`: 512 → 256 点，输出 128 维特征
   - `sa2`: 256 → 128 点，输出 256 维特征
   - `sa3`: 128 → 64 点，输出 512 维特征

2. **解码器（Decoder）- Feature Propagation 层：**
   - `fp3`: 从 64 点传播到 128 点
   - `fp2`: 从 128 点传播到 256 点
   - `fp1`: 从 256 点传播到 512 点
   - `fp0`: 从 512 点传播到原始点数（2048）

3. **分类头：**
   - `conv1`: Conv1d(128+16, 128)
   - `conv2`: Conv1d(128, num_class)

---

## 二、前向传播 (`forward`)

### 输入参数

```python
def forward(self, xyz, cls_label)
```

#### 1. `xyz` - 点云数据
- **维度格式**: `[B, N, C]`
- **说明**:
  - `B`: Batch size（批次大小）
  - `N`: 点的数量，通常为 **2048**（ShapeNet 数据集）
  - `C`: 通道数
    - 如果 `normal_channel=True`: `C = 6`（前3维为坐标 xyz，后3维为法向量 normals）
    - 如果 `normal_channel=False`: `C = 3`（仅坐标 xyz）

**示例：**
```python
# 使用法向量
xyz = torch.randn(16, 2048, 6)  # [B=16, N=2048, C=6]
# 不使用法向量
xyz = torch.randn(16, 2048, 3)  # [B=16, N=2048, C=3]
```

#### 2. `cls_label` - 类别标签
- **维度格式**: `[B]`
- **数据类型**: `torch.long`
- **取值范围**: `0-15`（ShapeNet 有 16 个物体类别）
- **说明**: 每个样本的物体类别标签（如：Airplane=0, Chair=12 等）

**示例：**
```python
cls_label = torch.tensor([0, 12, 3, 5, ...])  # [B=16]
```

---

### 前向传播流程

#### 步骤 1: 分离坐标和法向量
```python
if self.normal_channel:
    norm = xyz[:, :, 3:]  # [B, N, 3] - 法向量
    xyz = xyz[:, :, :3]   # [B, N, 3] - 坐标
```

#### 步骤 2: 编码器（下采样）
```python
# 输入: xyz [B, 2048, 3], norm [B, 2048, 3]
l0_xyz, l0_norm, l0_points = self.sa0(xyz, norm, None)
# 输出: l0_xyz [B, 512, 3], l0_norm [B, 512, 3], l0_points [B, 64, 512]

l1_xyz, l1_norm, l1_points = self.sa1(l0_xyz, l0_norm, l0_points)
# 输出: l1_xyz [B, 256, 3], l1_norm [B, 256, 3], l1_points [B, 128, 256]

l2_xyz, l2_norm, l2_points = self.sa2(l1_xyz, l1_norm, l1_points)
# 输出: l2_xyz [B, 128, 3], l2_norm [B, 128, 3], l2_points [B, 256, 128]

l3_xyz, l3_norm, l3_points = self.sa3(l2_xyz, l2_norm, l2_points)
# 输出: l3_xyz [B, 64, 3], l3_norm [B, 64, 3], l3_points [B, 512, 64]
```

**Set Abstraction 层输出格式：**
- `xyz`: `[B, npoint, 3]` - 采样后的点坐标
- `norm`: `[B, npoint, 3]` - 采样后的法向量
- `points`: `[B, out_channel, npoint]` - 特征（注意：通道维度在中间）

#### 步骤 3: 解码器（上采样）
```python
# 从 l3 传播到 l2
l2_points = self.fp3(l2_xyz, l3_xyz, l2_norm, l3_norm, l2_points, l3_points)
# 输入: l2_points [B, 256, 128], l3_points [B, 512, 64]
# 输出: l2_points [B, 512, 128]

# 从 l2 传播到 l1
l1_points = self.fp2(l1_xyz, l2_xyz, l1_norm, l2_norm, l1_points, l2_points)
# 输入: l1_points [B, 128, 256], l2_points [B, 512, 128]
# 输出: l1_points [B, 512, 256]

# 从 l1 传播到 l0
l0_points = self.fp1(l0_xyz, l1_xyz, l0_norm, l1_norm, l0_points, l1_points)
# 输入: l0_points [B, 64, 512], l1_points [B, 512, 256]
# 输出: l0_points [B, 256, 512]
```

#### 步骤 4: 融合类别信息
```python
# 将类别标签转换为 one-hot 编码并扩展到所有点
cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)  # [B, 16, N]
# 从 l0 传播到原始点云
l0_points = self.fp0(xyz, l0_xyz, norm, l0_norm, cls_label_one_hot, l0_points)
# 输入: xyz [B, 2048, 3], l0_points [B, 256, 512], cls_label_one_hot [B, 16, 2048]
# 输出: l0_points [B, 128, 2048]
```

#### 步骤 5: 分类头
```python
# 拼接类别信息
l0_points = torch.cat([l0_points, cls_label_one_hot], dim=1)  # [B, 128+16, 2048]
# 卷积层
feat = F.relu(self.bn1(self.conv1(l0_points)))  # [B, 128, 2048]
x = self.drop1(feat)  # [B, 128, 2048]
x = self.conv2(x)  # [B, num_class, 2048]
x = F.log_softmax(x, dim=1)  # [B, num_class, 2048]
x = x.permute(0, 2, 1)  # [B, 2048, num_class]
```

---

### 输出结果

#### 返回值

```python
return x, l3_points
```

#### 1. `x` - 分割预测结果
- **维度格式**: `[B, N, num_class]`
- **说明**:
  - `B`: Batch size
  - `N`: 点的数量（2048）
  - `num_class`: 分割类别数（50）
- **数值范围**: 经过 `log_softmax`，值为负对数概率
- **用途**: 每个点的部件类别预测

**示例：**
```python
x.shape  # torch.Size([16, 2048, 50])
# x[i, j, k] 表示第 i 个样本的第 j 个点属于第 k 个部件类别的 log 概率
```

#### 2. `l3_points` - 最深层特征
- **维度格式**: `[B, 512, 64]`
- **说明**:
  - `B`: Batch size
  - `512`: 特征通道数
  - `64`: 采样点数（最深层）
- **用途**: 可用于可视化或其他下游任务

---

## 三、完整数据流示例

### 输入示例
```python
# 批次大小 B=16，点数 N=2048
xyz = torch.randn(16, 2048, 6).cuda()  # [B, N, 6] - 包含坐标和法向量
cls_label = torch.randint(0, 16, (16,)).cuda()  # [B] - 类别标签
```

### 模型调用
```python
model = get_model(num_class=50, normal_channel=True).cuda()
seg_pred, trans_feat = model(xyz, cls_label)
```

### 输出示例
```python
seg_pred.shape  # torch.Size([16, 2048, 50])
trans_feat.shape  # torch.Size([16, 512, 64])

# 获取每个点的预测类别
pred_choice = seg_pred.argmax(dim=-1)  # [16, 2048]
```

---

## 四、维度变化总结

| 阶段 | 操作 | 输入维度 | 输出维度 |
|------|------|----------|----------|
| **输入** | - | `[B, 2048, 6]` | - |
| **分离** | 坐标/法向量分离 | `[B, 2048, 6]` | `xyz: [B, 2048, 3]`, `norm: [B, 2048, 3]` |
| **SA0** | Set Abstraction | `xyz: [B, 2048, 3]` | `xyz: [B, 512, 3]`, `points: [B, 64, 512]` |
| **SA1** | Set Abstraction | `points: [B, 64, 512]` | `xyz: [B, 256, 3]`, `points: [B, 128, 256]` |
| **SA2** | Set Abstraction | `points: [B, 128, 256]` | `xyz: [B, 128, 3]`, `points: [B, 256, 128]` |
| **SA3** | Set Abstraction | `points: [B, 256, 128]` | `xyz: [B, 64, 3]`, `points: [B, 512, 64]` |
| **FP3** | Feature Propagation | `points: [B, 512, 64]` | `points: [B, 512, 128]` |
| **FP2** | Feature Propagation | `points: [B, 512, 128]` | `points: [B, 512, 256]` |
| **FP1** | Feature Propagation | `points: [B, 512, 256]` | `points: [B, 256, 512]` |
| **FP0** | Feature Propagation | `points: [B, 256, 512]` | `points: [B, 128, 2048]` |
| **分类头** | Conv + Softmax | `points: [B, 128, 2048]` | `pred: [B, 2048, 50]` |

---

## 五、注意事项

1. **坐标归一化**: 输入的点云坐标应该在数据加载器中进行归一化（去中心化并缩放到单位球内）

2. **法向量**: 如果使用法向量，需要确保法向量与坐标同时进行旋转等变换，保持一致性

3. **类别标签**: `cls_label` 必须是 0-15 之间的整数，对应 ShapeNet 的 16 个物体类别

4. **特征维度**: Set Abstraction 层输出的特征维度是 `[B, C, N]`（通道在中间），而最终输出是 `[B, N, C]`（通道在最后）

5. **损失函数**: 输出经过 `log_softmax`，因此损失函数应使用 `NLLLoss` 或 `CrossEntropyLoss`

---

## 六、训练时的使用

在训练脚本中的使用方式：

```python
# 数据加载
points, label, target = trainDataLoader  # points: [B, N, 6], label: [B], target: [B, N]

# 模型前向传播
seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
# seg_pred: [B, N, 50], trans_feat: [B, 512, 64]

# 计算损失
seg_pred = seg_pred.contiguous().view(-1, num_part)  # [B*N, 50]
target = target.view(-1, 1)[:, 0]  # [B*N]
loss = criterion(seg_pred, target, trans_feat)
```

