"""
Titanic生存预测 - XGBoost完整方案
目标：达到85%+准确率
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 安装XGBoost（如果没有）
# !pip install xgboost

import xgboost as xgb
from xgboost import XGBClassifier

print("XGBoost版本:", xgb.__version__)

# ============================================================
# 第一步：数据加载和特征工程
# ============================================================
print("\n" + "="*70)
print("步骤1: 数据加载和特征工程")
print("="*70)

df = sns.load_dataset('titanic')
print(f"原始数据: {df.shape}")

# 1.1 提取称谓（Title）
df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 合并稀有称谓
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss', 'Don': 'Rare',
    'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
    'Sir': 'Rare', 'Capt': 'Rare'
}
df['title'] = df['title'].map(title_mapping).fillna('Rare')
print(f"✓ Title特征: {df['title'].unique()}")

# 1.2 家庭规模特征
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

# 家庭规模分类
df['family_category'] = pd.cut(df['family_size'], 
                                bins=[0, 1, 4, 20], 
                                labels=['Alone', 'Small', 'Large'])
print(f"✓ 家庭特征: family_size, is_alone, family_category")

# 1.3 票价分箱
df['fare_bin'] = pd.qcut(df['fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                         duplicates='drop')

# 1.4 年龄分箱（填充前）
df['age_known'] = df['age'].notna().astype(int)

# 1.5 客舱首字母
df['deck_letter'] = df['deck'].astype(str).str[0]
df['has_cabin'] = df['deck'].notna().astype(int)

# 1.6 票号首字母
df['ticket_prefix'] = df['ticket'].str.extract('([A-Za-z]+)', expand=False)
df['ticket_prefix'] = df['ticket_prefix'].fillna('None')

print(f"✓ 额外特征: fare_bin, age_known, deck_letter, has_cabin, ticket_prefix")

# ============================================================
# 第二步：智能缺失值填充
# ============================================================
print("\n" + "="*70)
print("步骤2: 智能缺失值填充")
print("="*70)

# 2.1 年龄填充（基于Title和Pclass）
age_mapping = df.groupby(['title', 'pclass'])['age'].median()

def fill_age(row):
    if pd.isna(row['age']):
        return age_mapping.get((row['title'], row['pclass']), df['age'].median())
    return row['age']

df['age'] = df.apply(fill_age, axis=1)
print(f"✓ 年龄缺失: {df['age'].isna().sum()}")

# 年龄分箱
df['age_bin'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
                       labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# 2.2 登船港口填充
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
print(f"✓ 登船港口缺失: {df['embarked'].isna().sum()}")

# 2.3 票价填充
df['fare'].fillna(df['fare'].median(), inplace=True)
df['fare_bin'].fillna('Q2', inplace=True)

# ============================================================
# 第三步：特征编码（XGBoost友好）
# ============================================================
print("\n" + "="*70)
print("步骤3: 特征编码")
print("="*70)

# XGBoost可以直接处理数值，所以我们使用Label Encoding
categorical_cols = ['sex', 'embarked', 'title', 'family_category', 
                   'age_bin', 'fare_bin', 'deck_letter', 'ticket_prefix']

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"✓ 编码 {col}: {len(le.classes_)} 个类别")

# 删除不需要的原始列
drop_cols = ['name', 'ticket', 'cabin', 'deck', 'class', 'who', 
             'adult_male', 'alive', 'alone', 'embark_town']
df.drop([col for col in drop_cols if col in df.columns], axis=1, inplace=True)

print(f"\n最终特征: {df.columns.tolist()}")
print(f"数据形状: {df.shape}")

# ============================================================
# 第四步：准备训练数据
# ============================================================
print("\n" + "="*70)
print("步骤4: 准备训练数据")
print("="*70)

X = df.drop('survived', axis=1)
y = df['survived']

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集生还率: {y_train.mean():.2%}, 测试集生还率: {y_test.mean():.2%}")

# ============================================================
# 第五步：基础XGBoost模型
# ============================================================
print("\n" + "="*70)
print("步骤5: 基础XGBoost模型")
print("="*70)

# 创建基础模型
xgb_base = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# 交叉验证
cv_scores = cross_val_score(xgb_base, X_train, y_train, cv=5, scoring='accuracy')
print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 训练模型
xgb_base.fit(X_train, y_train)

# 评估
train_score = xgb_base.score(X_train, y_train)
test_score = xgb_base.score(X_test, y_test)

print(f"训练集准确率: {train_score:.4f}")
print(f"测试集准确率: {test_score:.4f}")

# ============================================================
# 第六步：超参数调优 ⭐⭐⭐
# ============================================================
print("\n" + "="*70)
print("步骤6: 超参数调优（这可能需要几分钟）")
print("="*70)

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

# 先进行粗调
param_grid_coarse = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

grid_search = GridSearchCV(
    xgb_model,
    param_grid_coarse,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n✓ 最佳参数: {grid_search.best_params_}")
print(f"✓ 最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳模型
best_xgb = grid_search.best_estimator_

# ============================================================
# 第七步：模型评估
# ============================================================
print("\n" + "="*70)
print("步骤7: 详细模型评估")
print("="*70)

# 预测
y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

# 最终分数
final_train_score = best_xgb.score(X_train, y_train)
final_test_score = best_xgb.score(X_test, y_test)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"训练集准确率: {final_train_score:.4f}")
print(f"测试集准确率: {final_test_score:.4f}")
print(f"ROC-AUC分数:  {roc_auc:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['未生还', '生还']))

# ============================================================
# 第八步：可视化结果
# ============================================================
print("\n" + "="*70)
print("步骤8: 可视化")
print("="*70)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 8.1 混淆矩阵
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['未生还', '生还'],
            yticklabels=['未生还', '生还'])
ax1.set_title('混淆矩阵', fontsize=12, fontweight='bold')
ax1.set_ylabel('真实')
ax1.set_xlabel('预测')

# 8.2 ROC曲线
ax2 = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax2.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlabel('假正例率')
ax2.set_ylabel('真正例率')
ax2.set_title('ROC曲线', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 8.3 特征重要性（Top 15）
ax3 = fig.add_subplot(gs[0, 2])
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False).head(15)

ax3.barh(range(len(importance_df)), importance_df['importance'], color='teal')
ax3.set_yticks(range(len(importance_df)))
ax3.set_yticklabels(importance_df['feature'])
ax3.set_xlabel('重要性')
ax3.set_title('Top 15 特征重要性', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# 8.4 学习曲线
ax4 = fig.add_subplot(gs[1, :])
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores_list = []
test_scores_list = []

for train_size in train_sizes:
    n_samples = int(len(X_train) * train_size)
    X_subset = X_train.iloc[:n_samples]
    y_subset = y_train.iloc[:n_samples]
    
    best_xgb.fit(X_subset, y_subset)
    train_scores_list.append(best_xgb.score(X_subset, y_subset))
    test_scores_list.append(best_xgb.score(X_test, y_test))

ax4.plot(train_sizes * len(X_train), train_scores_list, 'o-', 
         color='r', label='训练集')
ax4.plot(train_sizes * len(X_train), test_scores_list, 'o-', 
         color='g', label='测试集')
ax4.set_xlabel('训练样本数')
ax4.set_ylabel('准确率')
ax4.set_title('学习曲线', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 重新训练最佳模型
best_xgb.fit(X_train, y_train)

# 8.5 预测概率分布
ax5 = fig.add_subplot(gs[2, 0])
survived_proba = y_pred_proba[y_test == 1]
died_proba = y_pred_proba[y_test == 0]

ax5.hist(died_proba, bins=30, alpha=0.5, label='未生还', color='red')
ax5.hist(survived_proba, bins=30, alpha=0.5, label='生还', color='green')
ax5.set_xlabel('预测生还概率')
ax5.set_ylabel('频数')
ax5.set_title('预测概率分布', fontsize=12, fontweight='bold')
ax5.legend()
ax5.axvline(0.5, color='black', linestyle='--', linewidth=1)

# 8.6 性能对比
ax6 = fig.add_subplot(gs[2, 1:])
models_comparison = {
    '原始模型': 0.787,
    'XGBoost基础': test_score,
    'XGBoost优化': final_test_score
}

bars = ax6.bar(models_comparison.keys(), models_comparison.values(), 
               color=['lightcoral', 'skyblue', 'gold'])
ax6.set_ylabel('测试集准确率')
ax6.set_title('模型性能对比', fontsize=12, fontweight='bold')
ax6.set_ylim(0.7, 0.9)
ax6.grid(axis='y', alpha=0.3)

for bar, (name, score) in zip(bars, models_comparison.items()):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Titanic XGBoost模型完整分析', fontsize=16, fontweight='bold', y=0.995)
plt.show()

# ============================================================
# 第九步：总结
# ============================================================
print("\n" + "="*70)
print("优化总结")
print("="*70)

improvement = final_test_score - 0.787
print(f"📊 性能提升:")
print(f"   原始准确率:       78.7%")
print(f"   XGBoost基础:     {test_score*100:.1f}%")
print(f"   XGBoost优化:     {final_test_score*100:.1f}%")
print(f"   总提升:          {improvement*100:.1f}个百分点 ({improvement/0.787*100:.1f}%)")

print(f"\n🎯 关键成功因素:")
print(f"   ✓ 高级特征工程（Title、family_size等）")
print(f"   ✓ XGBoost算法优势")
print(f"   ✓ 超参数调优")
print(f"   ✓ 智能缺失值处理")

print(f"\n📈 模型指标:")
print(f"   准确率:    {final_test_score:.4f}")
print(f"   ROC-AUC:  {roc_auc:.4f}")
print(f"   过拟合检查: 训练{final_train_score:.3f} vs 测试{final_test_score:.3f}")

print(f"\n🔝 Top 5 重要特征:")
for idx, row in importance_df.head(5).iterrows():
    print(f"   {row['feature']:20s} {row['importance']:.4f}")

print(f"\n💡 进一步提升建议:")
print(f"   1. 特征选择（去除低重要性特征）")
print(f"   2. 模型融合（XGBoost + Random Forest + LightGBM）")
print(f"   3. 更细致的超参数调优（贝叶斯优化）")
print(f"   4. 增强数据（SMOTE处理类别不平衡）")
print(f"   5. 尝试深度学习方法")