import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv('D:\\Pycharm\\BERT\\realdata1.csv')
tm_threshold = 60
df['category'] = np.where(df['Tm Protein'] > tm_threshold, 'thermo', 'meso')
df['Tm'] = df['category'].replace({'thermo': 0, 'meso': 1})
df1 = df.drop(['Tm Protein', 'Protein_ID', 'category'], axis=1)

sequences = df1['Sequence'].tolist()
labels = df1['Tm'].values

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("biobert")
model = AutoModel.from_pretrained("biobert")

def get_bert_embeddings(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.numpy().flatten()

print("开始提取BERT嵌入向量，过程较慢，请耐心等待...")
embeddings = [get_bert_embeddings(seq) for seq in sequences]
X = np.array(embeddings)
y = labels

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 定义参数空间和折数
NUM_FOLDS = 10
param_grid = {
    'C': [6],
    'gamma': [0.001],
    'kernel': ['rbf']
}

kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

best_score = 0
best_params = None
mcc_scores =[]

print("开始交叉验证和网格搜索...")



for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    svc = SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_tr, y_tr)

    y_pred = grid_search.predict(X_val)

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    accuracy = accuracy_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)

    print(f"Fold {fold}:")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print("--------------------------")

    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

avg_mcc = np.mean(mcc_scores)

print(f"\navg_mcc：{avg_mcc:.4f}")
print(f"\n交叉验证选出的最佳参数：{best_params}")

# 用最佳参数训练整个训练集
final_model = SVC(**best_params)
final_model.fit(X_train, y_train)

# 测试集评估
y_test_pred = final_model.predict(X_test)
precision_test = precision_score(y_test, y_test_pred, zero_division=0)
recall_test = recall_score(y_test, y_test_pred, zero_division=0)
f1_test = f1_score(y_test, y_test_pred, zero_division=0)
accuracy_test = accuracy_score(y_test, y_test_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)

print("\n测试集评估结果：")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
print(f"MCC: {mcc_test:.4f}")
