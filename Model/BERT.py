import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_csv('../Data/data1.csv')

# 2. 分类标签生成
tm = 60
df['category'] = np.where(df['Tm Protein'] > tm, 'thermo', 'meso')
df['Tm'] = df['category'].replace(['thermo', 'meso'], [0, 1])
df1 = df.drop(['Tm Protein', 'Protein_ID', 'category'], axis=1)

# 3. 提取序列和标签
data = df1
sequences = data['Sequence']
Tm = data['Tm']

# 4. BERT 词向量提取函数
def get_bert_embeddings(sequence):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        sequence_embeddings = outputs.pooler_output
    return sequence_embeddings.numpy()

# 5. 执行 BERT 编码
embeddings = [get_bert_embeddings(seq) for seq in sequences]
embeddings_flattened = [emb.flatten() for emb in embeddings]
X = np.array(embeddings_flattened)
y = Tm.values

# 6. 训练集/测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 7. 定义参数网格
param_grid = {
    'C': [1],
    'gamma': [1],
    'kernel': ['linear']
}

# 8. 交叉验证初始化
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []
best_score = 0
fold = 1

# 9. 交叉验证训练
for train_index, val_index in skf.split(X_train, y_train):
    print(f"\nFold {fold}:")
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_fold, y_train_fold)

    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val_fold)

    accuracy = accuracy_score(y_val_fold, y_val_pred)
    precision = precision_score(y_val_fold, y_val_pred, zero_division=0)
    recall = recall_score(y_val_fold, y_val_pred, zero_division=0)
    f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)
    mcc = matthews_corrcoef(y_val_fold, y_val_pred)

    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")

    fold_results.append({
        'fold': fold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'params': grid_search.best_params_,
        'model': best_model
    })

    if accuracy > best_score:
        best_score = accuracy
        best_fold = fold
        best_params = grid_search.best_params_
        best_model_overall = best_model

    fold += 1

# 10. 输出最佳结果
print(f"\nBest Fold: {best_fold}")
print(f"Best Fold Parameters: {best_params}")
print(f"Best Fold Accuracy: {best_score:.4f}")

# 11. 在整个训练集上训练最终模型
final_model = SVC(**best_params)
final_model.fit(X_train, y_train)

# 12. 在测试集上评估
# 12. 在测试集上评估
y_test_pred = final_model.predict(X_test)

accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred, zero_division=0)
recall_test = recall_score(y_test, y_test_pred, zero_division=0)
f1_test = f1_score(y_test, y_test_pred, zero_division=0)
mcc_test = matthews_corrcoef(y_test, y_test_pred)
report_test = classification_report(y_test, y_test_pred, zero_division=0)

print("\nTest Set Results:")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
print(f"MCC: {mcc_test:.4f}")
print("Classification Report:")
print(report_test)
