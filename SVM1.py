import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# 读取数据
file_path = '../Data/data2.csv'
data = pd.read_csv(file_path)

# 选择目标值和特征值
X = data.iloc[:, 3:]
y = data.iloc[:, 2]

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义SVM模型和网格搜索参数
svm_model = SVC()
param_grid = {
    'C': [1,2,3,4,5,6,7,8,9,10],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}

# 网格搜索，使用10折交叉验证
grid_search = GridSearchCV(svm_model, param_grid, cv=10)
grid_search.fit(X_train, y_train)

# 最佳参数
best_params = grid_search.best_params_
print(f'\nBest Parameters: {best_params}')

# 使用最佳参数模型进行10折交叉验证，并输出每一折的结果
best_model = grid_search.best_estimator_

print("\nCross-validation results (10 folds):")
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(best_model, X_train, y_train, cv=10, scoring=scoring)

# 输出每一折的结果
for i in range(10):
    print(f"Fold {i+1}:")
    print(f"  Accuracy: {cv_results['test_accuracy'][i]:.4f}")
    print(f"  Precision: {cv_results['test_precision'][i]:.4f}")
    print(f"  Recall: {cv_results['test_recall'][i]:.4f}")
    print(f"  F1 Score: {cv_results['test_f1'][i]:.4f}")

# 在测试集上评估最终模型
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
report = classification_report(y_test, y_pred, zero_division=0)

print(f'\nTest Set Results:')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Classification Report:')
print(report)
