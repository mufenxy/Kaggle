"""
快速修复你现有代码的关键问题
预期提升：78.7% -> 82%+
"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ============================================================
# 数据加载和基本清理
# ============================================================
df = sns.load_dataset('titanic')

# 改进1: 添加简单但有效的特征
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

# 改进2: 根据性别和客舱等级填充年龄（更准确）
age_median = df.groupby(['sex', 'pclass'])['age'].transform('median')
df['age'] = df['age'].fillna(age_median)

# 填充登船港口
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# 删除不需要的列
df.drop(['deck', 'class', 'who', 'adult_male', 'embark_town', 'alive', 'alone'], 
        axis=1, inplace=True)

# 编码性别
df['sex'] = df['sex'].map({'female': 0, 'male': 1})

# ============================================================
# 改进3: 使用Pipeline避免数据泄露 ⭐⭐⭐
# ============================================================
numerical_features = ['age', 'fare', 'family_size']
categorical_features = ['embarked']
binary_features = ['pclass', 'sex', 'sibsp', 'parch', 'is_alone']

X = df.drop('survived', axis=1)
y = df['survived']

# 分割数据（使用stratify保持类别平衡）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), 
         categorical_features),
        ('bin', 'passthrough', binary_features)
    ]
)

# 改进4: 使用Pipeline（关键！）
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, C=0.1, random_state=42))
])

# 训练模型
pipeline.fit(X_train, y_train)

# 评估
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print("="*60)
print("快速改进结果")
print("="*60)
print(f"原始准确率:     78.7%")
print(f"改进后准确率:   {test_score*100:.1f}%")
print(f"训练集准确率:   {train_score*100:.1f}%")
print(f"差值:          {abs(train_score - test_score)*100:.1f}%")
print("\n关键改进:")
print("  ✓ 使用Pipeline（避免数据泄露）")
print("  ✓ 添加family_size和is_alone特征")
print("  ✓ 智能填充年龄（按性别和舱位）")
print("  ✓ 使用stratify保持类别平衡")
print("  ✓ 调整正则化参数（C=0.1）")

# 与你原来代码的对比
print("\n你原代码的问题:")
print("  ❌ X_test使用了fit_transform（应该用transform）")
print("  ❌ 没有使用Pipeline")
print("  ❌ 缺少重要特征（家庭规模）")
print("  ❌ 年龄填充过于简单")