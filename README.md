# UserProfile
hi, this is Sariel, and here is how to make a user profile



# 一、电商用户行为数据分析项目

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

# 二、电商用户RFM分群分析项目

## 项目概述
本项目基于电商平台的用户行为数据，采用RFM模型和K-means聚类算法，对1000名用户进行价值分群。通过构建Recency(近度)、Frequency(频度)、Monetary(价值)三维指标体系，识别出4类典型用户群体及其消费特征，为精准营销提供数据支持。

## 数据文件
- 文件名：`ecommerce_user_behavior_dataset.csv`
- 数据结构：
  - 第一行：英文列名
  - 第二行：中文解释
  - 第三行开始：实际数据
- 关键RFM指标：
  - Recency：`Last_Login_Days_Ago`（最近登录天数，取负值）
  - Frequency：`Purchase_Frequency`（购买频率）
  - Monetary：`Total_Spending`（总消费金额）

## 快速开始

### 环境要求
```bash
Python 3.8+
pandas==1.3.4
scikit-learn==1.0.2
plotly==5.10.0
numpy==1.21.5
```

### 数据加载与预处理
```python
import pandas as pd

# 读取数据（跳过前两行描述）
df = pd.read_csv('ecommerce_user_behavior_dataset.csv', 
                skiprows=2,
                header=None,
                names=['User_ID','Age','Gender','Location','Income',
                      'Interests','Last_Login_Days_Ago','Purchase_Frequency',
                      'Average_Order_Value','Total_Spending',
                      'Product_Category_Preference','Time_Spent_on_Site_Minutes',
                      'Pages_Viewed','Newsletter_Subscription'],
                encoding='gbk')
```

## 核心分析代码

### RFM指标计算
```python
rfm = df.groupby('User_ID').agg({
    'Last_Login_Days_Ago': 'min',
    'Purchase_Frequency': 'sum',
    'Total_Spending': 'sum'
}).rename(columns={
    'Last_Login_Days_Ago': 'Recency',
    'Purchase_Frequency': 'Frequency',
    'Total_Spending': 'Monetary'
})

# 处理缺失值
rfm['Recency'] = -rfm['Recency'].fillna(df['Last_Login_Days_Ago'].max()+30)
rfm['Frequency'] = rfm['Frequency'].fillna(0)
rfm['Monetary'] = rfm['Monetary'].fillna(0)
```

### K-means聚类分析
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 标准化处理
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# 聚类分析
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
```

## 用户分群结果

| 用户类型 | 数量 | Recency均值 | Frequency均值 | Monetary均值 | 特征描述 |
|---------|------|------------|--------------|-------------|---------|
| 高价值用户 | 260 | -15.88 | 7.08 | 3829.28 | 高活跃、高频次、高消费 |
| 潜力用户 | 269 | -6.68 | 4.05 | 1676.17 | 近期活跃、中等消费 |
| 一般保持用户 | 252 | -22.27 | 5.31 | 1257.25 | 低活跃、稳定消费 |
| 流失风险用户 | 219 | -18.48 | 1.66 | 3605.60 | 低活跃、低频次但高客单 |

## 可视化分析
1. RFM三维散点图：
```python
import plotly.express as px
fig = px.scatter_3d(rfm.reset_index(),
                   x='Recency', y='Frequency', z='Monetary',
                   color='User_Type')
fig.show()
```

2. 群体特征雷达图：
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cluster_means = rfm.groupby('User_Type')[['Recency','Frequency','Monetary']].mean()
cluster_means_scaled = scaler.fit_transform(cluster_means)
```

## 营销建议
1. **高价值用户**：提供VIP专属服务和高端产品推荐
2. **潜力用户**：通过优惠券和限时促销提升购买频次
3. **一般保持用户**：定期推送个性化内容保持活跃度
4. **流失风险用户**：设计大客户召回方案和专属优惠

## 后续优化方向
1. 结合更多维度优化分群模型（如产品偏好、浏览行为）
2. 建立动态用户分群监控系统
3. 设计A/B测试验证不同营销策略效果
4. 开发用户生命周期预测模型

## 许可
本项目采用MIT License
