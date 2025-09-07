from itertools import permutations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from backend.dependencies.milvus import milvus
from backend.settings import settings

validation_data = milvus.get_all_names()
print(f"Total items: {len(validation_data)}")

# Store TOP-K duplicate file paths
top_k_results = {}
for item in validation_data:
    results = milvus.search_data(item["embedding"])
    if results:
        top_k_results[item["file_path"]] = [
            res.entity.get("file_path") for res in results[0]
        ]
    else:
        top_k_results[item["file_path"]] = []

y_true = []
y_pred = []

for a, b in tqdm(
    permutations(validation_data, 2),
    total=len(validation_data) * (len(validation_data) - 1),
):
    # Ground Truth
    same_person = int(a["name"].lower() == b["name"].lower())
    y_true.append(same_person)

    # Prediction
    pred_same = int(b["file_path"] in top_k_results[a["file_path"]])
    y_pred.append(pred_same)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Evaluation Results")
print("------------------")
print(f"TOP-K: {settings.TOP_K}")
print(f"Distance Threshold: {settings.MILVUS_RADIUS}")
print("------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
