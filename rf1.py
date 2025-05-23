import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from math import sqrt
import numpy as np

# 读取数据
df = pd.read_csv("../Data/data1.csv")
array = df.values

# Features and target
X = array[:, 3:].astype("float")
y = array[:, 1].astype("float")

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 随机森林回归模型
rf = RandomForestRegressor(random_state=42)

# 网格搜索设置（如需要添加参数，可在此处定义）
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['squared_error', 'absolute_error']
}


# 网格搜索
grid_search = GridSearchCV(rf, param_grid, cv=10, scoring='r2')
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)


best_model = grid_search.best_estimator_

# 使用最佳模型进行10折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
print("Cross-validation results on training set (10 folds):")

# 初始化指标
train_r2_scores, train_mae_scores, train_rmse_scores, train_pcc_scores = [], [], [], []

for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # 训练并预测每一折
    best_model.fit(X_train_fold, y_train_fold)
    y_val_pred = best_model.predict(X_val_fold)

    # 计算并记录每一折的指标
    train_r2_scores.append(r2_score(y_val_fold, y_val_pred))
    train_mae_scores.append(mean_absolute_error(y_val_fold, y_val_pred))
    train_rmse_scores.append(sqrt(mean_squared_error(y_val_fold, y_val_pred)))
    train_pcc_scores.append(pearsonr(y_val_fold, y_val_pred)[0])

    # 输出每一折的结果
    print(f"Fold {fold} - R2: {train_r2_scores[-1]:.4f}, MAE: {train_mae_scores[-1]:.4f}, "
          f"RMSE: {train_rmse_scores[-1]:.4f}, PCC: {train_pcc_scores[-1]:.4f}")

# 输出训练集上10折交叉验证的平均结果
print("\nCross-validation mean results on training set:")
print(f"Mean R2: {np.mean(train_r2_scores):.4f}, Mean MAE: {np.mean(train_mae_scores):.4f}, "
      f"Mean RMSE: {np.mean(train_rmse_scores):.4f}, Mean PCC: {np.mean(train_pcc_scores):.4f}")

# 测试集上的最终评估
y_test_pred = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
test_pcc = pearsonr(y_test, y_test_pred)[0]

print("\nTest Set Results:")
print(f"Test R2: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, PCC: {test_pcc:.4f}")
