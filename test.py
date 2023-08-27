import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Giả sử bạn có mảng nhãn thực tế `y_true` và mảng dự đoán liên tục `y_pred`
y_true = torch.Tensor([[1, 0, 1, 0],   # Ảnh 1: "Chó" và "Hoa"
                   [0, 1, 1, 1],   # Ảnh 2: "Mèo", "Hoa", "Xe hơi"
                   [1, 0, 0, 0]])  # Ảnh 3: "Chó"
print(type(y_true[0][0]))
y_pred = torch.Tensor([[0.8, 0.2, 0.6, 0.1],  # Dự đoán cho ảnh 1
                   [0.8, 0.9, 0.7, 0.6],  # Dự đoán cho ảnh 2
                   [0.6, 0.3, 0.2, 0.4]]) # Dự đoán cho ảnh 3

# Chuyển đổi mảng dự đoán liên tục thành mảng nhãn dự đoán nhị phân bằng ngưỡng 0.5
y_pred_binary = (y_pred > 0.5).float()
print(type(y_pred_binary))
# print(y_pred_binary.)

print("Nhãn thực tế:")
print(y_true)
print("Nhãn dự đoán nhị phân:")
print(y_pred_binary)

# y_true = y_true.tolist()
# y_pred_binary = y_pred_binary.tolist()
correct = 0
correct += (y_pred_binary == y_true).float().sum()
print(correct/12)
# print((y_pred_binary == y_true).all(axis = (0, 1)).mean())

precision = precision_score(y_true, y_pred_binary, average=None)
recall = recall_score(y_true, y_pred_binary, average=None)
auc_score = roc_auc_score(y_true, y_pred, multi_class='ovr')
print("Precision:", precision)
print("Recall:", recall)
print("AUC score:", auc_score)

