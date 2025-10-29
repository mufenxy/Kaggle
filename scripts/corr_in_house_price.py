"""
房价数据相关性检验
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 数值特征与目标变量的相关性 ============
# 只选择数值型列
numerical_data = data_train.select_dtypes(include='number')

# 计算与 SalePrice 的相关系数
correlations = numerical_data.corr()['SalePrice'].sort_values(ascending=False)

print("=" * 50)
print("与房价相关性最高的前15个特征：")
print("=" * 50)
print(correlations.head(15))

print("\n" + "=" * 50)
print("与房价相关性最低的15个特征：")
print("=" * 50)
print(correlations.tail(15))

# ============ 2. 相关性热力图 ============
# 选择相关性最高的10个特征绘制热力图
top_features = correlations.head(11).index  # 包括 SalePrice 自己

plt.figure(figsize=(12, 10))
sns.heatmap(
    numerical_data[top_features].corr(), 
    annot=True,  # 显示数值
    fmt='.2f',   # 保留两位小数
    cmap='coolwarm',  # 颜色方案
    center=0,    # 以0为中心
    square=True,
    linewidths=0.5
)
plt.title('Top 10 特征相关性热力图', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============ 3. 特征之间的相关性（检测多重共线性）============
# 找出高度相关的特征对（相关系数 > 0.8）
print("\n" + "=" * 50)
print("高度相关的特征对（|r| > 0.8）：")
print("=" * 50)

corr_matrix = numerical_data.corr().abs()
# 设置对角线和下三角为0，避免重复
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr = []
for column in upper_triangle.columns:
    high_corr_features = upper_triangle[column][upper_triangle[column] > 0.8]
    if len(high_corr_features) > 0:
        for idx in high_corr_features.index:
            high_corr.append((column, idx, upper_triangle.loc[idx, column]))

for feat1, feat2, corr_val in high_corr:
    print(f"{feat1} <-> {feat2}: {corr_val:.3f}")

# ============ 4. 散点图矩阵（可视化关系）============
# 选择相关性最高的几个特征
top_features_for_plot = correlations.head(6).index  # 前5个+SalePrice

plt.figure(figsize=(15, 12))
pd.plotting.scatter_matrix(
    numerical_data[top_features_for_plot],
    figsize=(15, 12),
    diagonal='hist',  # 对角线显示直方图
    alpha=0.5,
    s=20
)
plt.suptitle('Top 5 特征散点图矩阵', fontsize=16, y=1.0)
plt.tight_layout()
plt.savefig('scatter_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============ 5. 单个特征的详细分析 ============
# 对相关性最高的几个特征进行详细分析
print("\n" + "=" * 50)
print("Top 5 特征的统计检验：")
print("=" * 50)

top_5_features = correlations.drop('SalePrice').head(5)

for feature in top_5_features.index:
    # Pearson 相关系数（线性相关）
    pearson_r, pearson_p = pearsonr(
        data_train[feature].dropna(), 
        data_train.loc[data_train[feature].notna(), 'SalePrice']
    )
    
    # Spearman 相关系数（单调相关）
    spearman_r, spearman_p = spearmanr(
        data_train[feature].dropna(), 
        data_train.loc[data_train[feature].notna(), 'SalePrice']
    )
    
    print(f"\n{feature}:")
    print(f"  Pearson r: {pearson_r:.4f}, p-value: {pearson_p:.4e}")
    print(f"  Spearman ρ: {spearman_r:.4f}, p-value: {spearman_p:.4e}")

# ============ 6. 分类特征的关系（箱线图）============
# 选择几个重要的分类特征
categorical_features = ['OverallQual', 'ExterQual', 'KitchenQual', 'BsmtQual']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, feature in enumerate(categorical_features):
    if feature in data_train.columns:
        data_train.boxplot(column='SalePrice', by=feature, ax=axes[idx])
        axes[idx].set_title(f'{feature} vs SalePrice')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('SalePrice')

plt.suptitle('分类特征与房价的关系', fontsize=16, y=1.0)
plt.tight_layout()
plt.savefig('categorical_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============ 7. 特征选择建议 ============
print("\n" + "=" * 50)
print("特征选择建议：")
print("=" * 50)

# 高相关特征（|r| > 0.5）
high_corr_features = correlations[abs(correlations) > 0.5].drop('SalePrice')
print(f"\n强相关特征（推荐保留）: {len(high_corr_features)} 个")
print(high_corr_features)

# 低相关特征（|r| < 0.1）
low_corr_features = correlations[abs(correlations) < 0.1]
print(f"\n弱相关特征（可考虑删除）: {len(low_corr_features)} 个")
print(low_corr_features)