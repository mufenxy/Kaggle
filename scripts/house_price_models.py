"""
房价预测：线性回归 vs XGBoost vs LightGBM
包含数据预处理、模型训练、性能对比、特征重要性分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 数据加载与预处理 ============
print("=" * 60)
print("步骤 1: 数据加载与预处理")
print("=" * 60)

# 读取数据
data_train = pd.read_csv('D:/code/kaggle/datasets/house_prices_advanced/train.csv')
data_test = pd.read_csv('D:/code/kaggle/datasets/house_prices_advanced/test.csv')

print(f"训练集形状: {data_train.shape}")
print(f"测试集形状: {data_test.shape}")

# 删除高缺失率特征
drop_list = ['LotFrontage', 'Alley', 'MasVnrType', 'FireplaceQu', 
             'PoolQC', 'Fence', 'MiscFeature']
data_train = data_train.drop(columns=drop_list, axis=1)
data_test = data_test.drop(columns=drop_list, axis=1)

# 处理剩余缺失值
# 数值型：用中位数填充
numerical_cols = data_train.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if data_train[col].isna().sum() > 0:
        median_val = data_train[col].median()
        data_train[col].fillna(median_val, inplace=True)
        if col in data_test.columns:
            data_test[col].fillna(median_val, inplace=True)

# 分类型：用众数填充
categorical_cols = data_train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if data_train[col].isna().sum() > 0:
        mode_val = data_train[col].mode()[0]
        data_train[col].fillna(mode_val, inplace=True)
        if col in data_test.columns:
            data_test[col].fillna(mode_val, inplace=True)

print(f"\n清洗后训练集缺失值数量: {data_train.isna().sum().sum()}")
print(f"清洗后测试集缺失值数量: {data_test.isna().sum().sum()}")

# ============ 2. 特征工程 ============
print("\n" + "=" * 60)
print("步骤 2: 特征工程")
print("=" * 60)

# 保存测试集ID
test_ids = data_test['Id']

# 分离特征和目标
X = data_train.drop(['Id', 'SalePrice'], axis=1)
y = data_train['SalePrice']
X_test = data_test.drop(['Id'], axis=1)

# 对目标变量进行log变换（减少偏斜）
y_log = np.log1p(y)

print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")

# 识别数值和分类特征
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"数值特征: {len(numerical_features)} 个")
print(f"分类特征: {len(categorical_features)} 个")

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_features),  # RobustScaler对异常值更稳健
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}")

# ============ 3. 定义评估函数 ============
def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """
    评估模型性能
    """
    # 训练集预测
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 验证集预测
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n{model_name} 性能:")
    print(f"  训练集 - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  验证集 - RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'model': model
    }

# ============ 4. 线性回归模型 ============
print("\n" + "=" * 60)
print("步骤 3: 训练线性回归模型")
print("=" * 60)

results = []

# 4.1 普通线性回归
print("\n训练普通线性回归...")
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
results.append(evaluate_model(lr_pipeline, X_train, X_val, y_train, y_val, "Linear Regression"))

# 4.2 Ridge回归（L2正则化）
print("\n训练Ridge回归...")
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=10.0))
])
ridge_pipeline.fit(X_train, y_train)
results.append(evaluate_model(ridge_pipeline, X_train, X_val, y_train, y_val, "Ridge Regression"))

# 4.3 Lasso回归（L1正则化）
print("\n训练Lasso回归...")
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.001, max_iter=10000))
])
lasso_pipeline.fit(X_train, y_train)
results.append(evaluate_model(lasso_pipeline, X_train, X_val, y_train, y_val, "Lasso Regression"))

# 4.4 ElasticNet（L1+L2正则化）
print("\n训练ElasticNet...")
elastic_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))
])
elastic_pipeline.fit(X_train, y_train)
results.append(evaluate_model(elastic_pipeline, X_train, X_val, y_train, y_val, "ElasticNet"))

# ============ 5. XGBoost模型 ============
print("\n" + "=" * 60)
print("步骤 4: 训练XGBoost模型")
print("=" * 60)

# 预处理数据（XGBoost需要数值化的数据）
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# 5.1 基础XGBoost
print("\n训练基础XGBoost...")
xgb_base = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    n_jobs=-1
)
xgb_base.fit(
    X_train_processed, y_train,
    eval_set=[(X_val_processed, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

class XGBPipelineWrapper:
    """包装XGBoost使其可以处理原始数据"""
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
    
    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

xgb_wrapper = XGBPipelineWrapper(preprocessor, xgb_base)
results.append(evaluate_model(xgb_wrapper, X_train, X_val, y_train, y_val, "XGBoost"))

# 5.2 调优XGBoost（可选，耗时较长）
print("\n训练调优XGBoost（网格搜索）...")
xgb_tuned = xgb.XGBRegressor(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

# 注意：这里为了速度只搜索部分参数，实际可以扩展
grid_search = GridSearchCV(
    xgb_tuned, 
    param_grid={'n_estimators': [500], 'learning_rate': [0.05], 'max_depth': [4]},
    cv=3, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_processed, y_train)

xgb_tuned_wrapper = XGBPipelineWrapper(preprocessor, grid_search.best_estimator_)
results.append(evaluate_model(xgb_tuned_wrapper, X_train, X_val, y_train, y_val, "XGBoost Tuned"))

# ============ 6. LightGBM模型 ============
print("\n" + "=" * 60)
print("步骤 5: 训练LightGBM模型")
print("=" * 60)

# 6.1 基础LightGBM
print("\n训练基础LightGBM...")
lgb_base = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_base.fit(
    X_train_processed, y_train,
    eval_set=[(X_val_processed, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

lgb_wrapper = XGBPipelineWrapper(preprocessor, lgb_base)
results.append(evaluate_model(lgb_wrapper, X_train, X_val, y_train, y_val, "LightGBM"))

# 6.2 调优LightGBM
print("\n训练调优LightGBM...")
lgb_tuned = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=50,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_tuned.fit(
    X_train_processed, y_train,
    eval_set=[(X_val_processed, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

lgb_tuned_wrapper = XGBPipelineWrapper(preprocessor, lgb_tuned)
results.append(evaluate_model(lgb_tuned_wrapper, X_train, X_val, y_train, y_val, "LightGBM Tuned"))

# ============ 7. 模型对比可视化 ============
print("\n" + "=" * 60)
print("步骤 6: 模型性能对比")
print("=" * 60)

results_df = pd.DataFrame(results)
print("\n所有模型性能对比:")
print(results_df[['model_name', 'val_rmse', 'val_mae', 'val_r2']].to_string(index=False))

# 7.1 性能对比柱状图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE对比
axes[0].barh(results_df['model_name'], results_df['val_rmse'], color='skyblue')
axes[0].set_xlabel('RMSE (验证集)')
axes[0].set_title('模型RMSE对比（越小越好）')
axes[0].invert_yaxis()

# MAE对比
axes[1].barh(results_df['model_name'], results_df['val_mae'], color='lightcoral')
axes[1].set_xlabel('MAE (验证集)')
axes[1].set_title('模型MAE对比（越小越好）')
axes[1].invert_yaxis()

# R²对比
axes[2].barh(results_df['model_name'], results_df['val_r2'], color='lightgreen')
axes[2].set_xlabel('R² Score (验证集)')
axes[2].set_title('模型R²对比（越大越好）')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 7.2 训练集vs验证集性能对比
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['train_rmse'], width, label='训练集', alpha=0.8)
bars2 = ax.bar(x + width/2, results_df['val_rmse'], width, label='验证集', alpha=0.8)

ax.set_xlabel('模型')
ax.set_ylabel('RMSE')
ax.set_title('训练集vs验证集RMSE对比（检测过拟合）')
ax.set_xticks(x)
ax.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('overfitting_check.png', dpi=300, bbox_inches='tight')
plt.show()

# ============ 8. 特征重要性分析 ============
print("\n" + "=" * 60)
print("步骤 7: 特征重要性分析")
print("=" * 60)

# XGBoost特征重要性
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 8.1 XGBoost
xgb_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(X_train_processed.shape[1])],
    'importance': xgb_base.feature_importances_
}).sort_values('importance', ascending=False).head(20)

axes[0].barh(range(20), xgb_importance['importance'], color='skyblue')
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(xgb_importance['feature'])
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('XGBoost Top 20 特征重要性')
axes[0].invert_yaxis()

# 8.2 LightGBM
lgb_importance = pd.DataFrame({
    'feature': [f'feature_{i}' for i in range(X_train_processed.shape[1])],
    'importance': lgb_tuned.feature_importances_
}).sort_values('importance', ascending=False).head(20)

axes[1].barh(range(20), lgb_importance['importance'], color='lightcoral')
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(lgb_importance['feature'])
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('LightGBM Top 20 特征重要性')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============ 9. 最佳模型预测 ============
print("\n" + "=" * 60)
print("步骤 8: 使用最佳模型生成提交文件")
print("=" * 60)

# 找到最佳模型（验证集RMSE最小）
best_model_idx = results_df['val_rmse'].idxmin()
best_model_info = results_df.loc[best_model_idx]
best_model = best_model_info['model']

print(f"\n最佳模型: {best_model_info['model_name']}")
print(f"验证集RMSE: {best_model_info['val_rmse']:.4f}")
print(f"验证集R²: {best_model_info['val_r2']:.4f}")

# 预测测试集
test_predictions_log = best_model.predict(X_test)
test_predictions = np.expm1(test_predictions_log)  # 逆log变换

# 生成提交文件
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("\n提交文件已保存为 'submission.csv'")

# ============ 10. 总结与建议 ============
