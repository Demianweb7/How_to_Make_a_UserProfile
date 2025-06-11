# UserProfile
hi, this is Sariel, and here is how to make a user profile



# 电商用户行为数据分析项目

## 项目概述
本项目对电商平台的用户行为数据进行分析，构建精细化的用户画像，包含1000条用户记录和15个特征维度。通过数据探查和可视化分析，识别了4类典型用户群体及其行为特征。

## 数据文件
- 文件名：`电商用户行为数据集.csv`
- 数据结构：
  - 第一行：英文列名
  - 第二行：中文解释
  - 第三行开始：实际数据
- 主要字段：
  - 用户属性：Age, Gender, Location, Income
  - 行为特征：Purchase_Frequency, Average_Order_Value, Total_Spending
  - 兴趣偏好：Interests, Product_Category_Preference
  - 互动指标：Time_Spent_on_Site_Minutes, Pages_Viewed

## 快速开始

### 环境要求
```bash
Python 3.8+
pandas==1.3.4
matplotlib==3.4.3
seaborn==0.11.2
```

### 数据加载
```python
import pandas as pd

# 正确读取方式（跳过前两行描述）
df = pd.read_csv('电商用户行为数据集.csv', 
                header=1,  # 使用第二行作为列名
                skiprows=[0],  # 跳过第一行
                encoding='utf-8')
```

## 分析代码

### 基础统计
```python
# 数值型变量描述统计
numeric_cols = ['Age', 'Income', 'Total_Spending']
print(df[numeric_cols].describe())

# 分类变量统计
categorical_cols = ['Gender', 'Location', 'Product_Category_Preference']
for col in categorical_cols:
    print(f"\n{col}分布：")
    print(df[col].value_counts(normalize=True))
```

### 关键可视化
1. 用户年龄分布：
```python
import seaborn as sns
sns.histplot(df['Age'], bins=30, kde=True)
```

2. 收入-消费关系：
```python
sns.scatterplot(x='Income', y='Total_Spending', hue='Gender', data=df)
```

3. 产品偏好分析：
```python
df['Product_Category_Preference'].value_counts().head(10).plot(kind='barh')
```

## 用户画像摘要

| 用户类型 | 核心特征 | 消费特点 | 典型场景 |
|---------|---------|---------|---------|
| 都市科技新贵 | 25-35岁，高收入，城市 | 高频高客单，偏好电子产品 | 追求新品，研究参数 |
| 郊区时尚主妇 | 30-45岁，中等收入 | 中频消费，服饰类为主 | 夜间浏览穿搭内容 |
| 农村务实消费者 | 40-55岁，中低收入 | 低频高客单，生活必需品 | 直接搜索按销量排序 |
| 银发品质生活家 | 55-65岁，收入两极 | 健康品类复购率高 | 电话咨询后购买 |

## 分析结论
1. 城市用户偏好电子/时尚品类，农村用户聚焦生活必需品
2. 25-34岁用户消费力最强，55+用户健康消费增长快
3. 营销订阅用户比非订阅用户客单价高42%
4. 收入与消费金额正相关，但存在高收入低消费群体

## 后续建议
1. 针对不同群体设计差异化营销策略
2. 优化搜索算法和产品推荐逻辑
3. 加强用户生命周期管理
4. 深化RFM模型分析

## 许可
本项目采用MIT License
