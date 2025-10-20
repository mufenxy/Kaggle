"""
Titanic XGBoost - 健壮版本
自动检测可用列，适配不同数据源
"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimHei']

# 安装XGBoost（如果需要）
try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    print(f"✓ XGBoost版本: {xgb.__version__}")
except ImportError:
    print("❌ 请先安装XGBoost: pip install xgboost")
    exit()

# ============================================================
# 第一步：数据加载和列检测
# ============================================================
print("\n" + "="*70)
print("步骤1: 数据加载和列检测")
print("="*70)

df = sns.load_dataset('titanic')
print(f"原始数据: {df.shape}")
print(f"可用列: {df.columns.tolist()}")

# 检查必需列
required_cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"❌ 缺少必需列: {missing_cols}")
    exit()

print(f"✓ 所有必需列都存在")

# ============================================================
# 第二步：智能特征工程（根据可用列）
# ============================================================
print("\n" + "="*70)
print("步骤2: 智能特征工程")
print("="*70)

# 2.1 家庭规模特征（基础）
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
print(f"✓ 创建family_size和is_alone")

# 2.2 家庭规模分类
df['family_category'] = pd.cut(df['family_size'], 
                                bins=[0, 1, 4, 20], 
                                labels=['Alone', 'Small', 'Large'])
print(f"✓ 创建family_category")

# 2.3 客舱特征（如果有cabin列）
if 'cabin' in df.columns:
    df['has_cabin'] = df['cabin'].notna().astype(int)
    print(f"✓ 创建has_cabin")
else:
    df['has_cabin'] = 0
    print(f"⚠️  没有cabin列，has_cabin设为0")

# 2.4 甲板特征（如果有deck列）
if 'deck' in df.columns:
    df['deck_letter'] = df['deck'].astype(str).str[0]
    df['deck_letter'] = df['deck_letter'].replace('n', 'Unknown')
    print(f"✓ 创建deck_letter")
else:
    df['deck_letter'] = 'Unknown'
    print(f"⚠️  没有deck列，deck_letter设为Unknown")

# 2.5 年龄相关特征
df['age_known'] = df['age'].notna().astype(int)
print(f"✓ 创建age_known")

# 2.6 票价分组
df['fare_bin'] = pd.qcut(df['fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                         duplicates='drop')
print(f"✓ 创建fare_bin")

# 2.7 性别×舱位交互特征
df['sex_pclass'] = df['sex'].astype(str) + '_' + df['pclass'].astype(str)
print(f"✓ 创建sex_pclass交互特征")

# ============================================================
# 第三步：智能缺失值填充
# ============================================================
print("\n" + "="*70)
print("步骤3: 缺失值处理")
print("="*70)

# 3.1 年龄填充（根据性别和舱位）
print(f"年龄缺失数: {df['age'].isna().sum()}")
if df['age'].isna().sum() > 0:
    age_median = df.groupby(['sex', 'pclass'])['age'].transform('median')
    df['age'] = df['age'].fillna(age_median)
    print(f"✓ 填充后年龄缺失: {df['age'].isna().sum()}")

# 3.2 年龄分组
df['age_bin'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                       labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
print(f"✓ 创建age_bin")

# 3.3 登船港口填充
if 'embarked' in df.columns:
    print(f"登船港口缺失数: {df['embarked'].isna().sum()}")
    if df['embarked'].isna().sum() > 0:
        df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
        print(f"✓ 填充后embarked缺失: {df['embarked'].isna().sum()}")
else:
    df['embarked'] = 'S'
    print(f"⚠️  没有embarked列，设为默认值'S'")

# 3.4 票价填充
if df['fare'].isna().sum() > 0:
    df['fare'].fillna(df['fare'].median(), inplace=True)
    df['fare_bin'].fillna('Q2', inplace=True)
    print(f"✓ 填充票价")

# ============================================================
# 第四步：特征编码
# ============================================================
print("\n" + "="*70)
print("步骤4: 特征编码")
print("="*70)

# 选择要使用的特征
feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
                'family_size', 'is_alone', 'has_cabin', 'age_known']

# 添加可用的分类特征
if 'embarked' in df.columns:
    feature_cols.append('embarked')
    
categorical_cols = ['sex', 'family_category', 'age_bin', 'fare_bin', 
                   'deck_letter', 'sex_pclass']
if 'embarked' in df.columns:
    categorical_cols.append('embarked')

# 将分类特征加入
for col in categorical_cols:
    if col in df.columns:
        feature_cols.append(col)

# Label Encoding
le_dict = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# 确保sex也被编码
if df['sex'].dtype == 'object':
    df['sex'] = LabelEncoder().fit_transform(df['sex'])

# 去重feature_cols
feature_cols = list(dict.fromkeys(feature_cols))

print(f"✓ 最终特征数: {len(feature_cols)}")
print(f"✓ 特征列表: {feature_cols}")

# ============================================================
# 第五步：准备训练数据
# ============================================================
print("\n" + "="*70)
print("步骤5: 准备训练数据")
print("="*70)

X = df[feature_cols].copy()
y = df['survived'].copy()

# 检查并处理任何剩余的缺失值
if X.isna().sum().sum() > 0:
    print(f"⚠️  发现剩余缺失值:")
    print(X.isna().sum()[X.isna().sum() > 0])
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    print(f"✓ 已填充所有缺失值")

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ 训练集: {X_train.shape}")
print(f"✓ 测试集: {X_test.shape}")
print(f"✓ 训练集生还率: {y_train.mean():.2%}")
print(f"✓ 测试集生还率: {y_test.mean():.2%}")

# ============================================================
# 第六步：训练XGBoost模型
# ============================================================
print("\n" + "="*70)
print("步骤6: 训练XGBoost模型")
print("="*70)

# 创建模型（优化参数）
xgb_model = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=1.5,
    random_state=42,
    eval_metric='logloss'
)

# 训练
print("训练中...")
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

# ============================================================
# 第七步：评估模型
# ============================================================
print("\n" + "="*70)
print("步骤7: 模型评估")
print("="*70)

# 预测
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# 计算指标
train_score = xgb_model.score(X_train, y_train)
test_score = xgb_model.score(X_test, y_test)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 交叉验证
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)

print(f"训练集准确率:   {train_score:.4f} ({train_score*100:.2f}%)")
print(f"测试集准确率:   {test_score:.4f} ({test_score*100:.2f}%)")
print(f"ROC-AUC分数:   {roc_auc:.4f}")
print(f"5折交叉验证:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"过拟合检查:    {abs(train_score - test_score):.4f}")

if abs(train_score - test_score) > 0.1:
    print("⚠️  警告: 可能存在轻微过拟合")
else:
    print("✓ 模型泛化良好")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['未生还', '生还']))

# ============================================================
# 第八步：特征重要性分析
# ============================================================
print("\n" + "="*70)
print("步骤8: 特征重要性")
print("="*70)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 重要特征:")
print(importance_df.head(10).to_string(index=False))

# ============================================================
# 第九步：可视化
# ============================================================
print("\n" + "="*70)
print("步骤9: 结果可视化")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 9.1 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
im = axes[0, 0].imshow(cm, cmap='Blues')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_xticklabels(['未生还', '生还'])
axes[0, 0].set_yticklabels(['未生还', '生还'])

for i in range(2):
    for j in range(2):
        text = axes[0, 0].text(j, i, cm[i, j],
                              ha="center", va="center", color="black",
                              fontsize=20, fontweight='bold')

axes[0, 0].set_title('混淆矩阵', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('真实标签')
axes[0, 0].set_xlabel('预测标签')

# 9.2 ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC (AUC = {roc_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='随机猜测')
axes[0, 1].set_xlabel('假正例率')
axes[0, 1].set_ylabel('真正例率')
axes[0, 1].set_title('ROC曲线', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 9.3 特征重要性（Top 10）
top_features = importance_df.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'], color='teal')
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_xlabel('重要性')
axes[1, 0].set_title('Top 10 特征重要性', fontsize=12, fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(axis='x', alpha=0.3)

# 9.4 模型对比
models = ['原始逻辑回归', 'XGBoost']
scores = [0.787, test_score]
bars = axes[1, 1].bar(models, scores, color=['lightcoral', 'gold'])
axes[1, 1].set_ylabel('测试集准确率')
axes[1, 1].set_title('模型性能对比', fontsize=12, fontweight='bold')
axes[1, 1].set_ylim(0.7, 0.9)
axes[1, 1].grid(axis='y', alpha=0.3)

for bar, score in zip(bars, scores):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('titanic_xgboost_results.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存为 'titanic_xgboost_results.png'")
plt.show()

# ============================================================
# 第十步：总结
# ============================================================
print("\n" + "="*70)
print("优化总结")
print("="*70)

improvement = test_score - 0.787
improvement_pct = improvement / 0.787 * 100

print(f"\n📊 性能对比:")
print(f"   原始逻辑回归:    78.7%")
print(f"   XGBoost:        {test_score*100:.1f}%")
print(f"   提升:           {improvement*100:.1f}个百分点 ({improvement_pct:.1f}%)")

print(f"\n🎯 关键指标:")
print(f"   测试集准确率:    {test_score:.4f}")
print(f"   ROC-AUC:        {roc_auc:.4f}")
print(f"   交叉验证:        {cv_scores.mean():.4f}")
print(f"   过拟合程度:      {abs(train_score - test_score):.4f}")

print(f"\n🔝 Top 5 重要特征:")
for idx, row in importance_df.head(5).iterrows():
    print(f"   {row['feature']:20s} {row['importance']:.4f}")

print(f"\n💡 关键成功因素:")
print(f"   ✓ 健壮的特征工程")
print(f"   ✓ XGBoost算法优势")
print(f"   ✓ 智能缺失值处理")
print(f"   ✓ 自动适配数据集")

print(f"\n🚀 进一步优化建议:")
print(f"   1. 超参数网格搜索（GridSearchCV）")
print(f"   2. 特征选择（去除低重要性特征）")
print(f"   3. 模型融合（Ensemble）")
print(f"   4. SMOTE处理类别不平衡")

print("\n" + "="*70)
print("✓ 完成!")
print("="*70)