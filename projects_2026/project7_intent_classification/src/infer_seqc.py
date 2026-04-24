import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "seqc_final_model")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "test.csv")


def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_input_text(text_src, text_tgt=""):
    src = safe_text(text_src)
    tgt = safe_text(text_tgt)

    if src and tgt:
        return src + " [SEP] " + tgt
    elif src:
        return src
    return tgt


def load_model():
    print("MODEL_PATH:", MODEL_PATH)
    print("TEST_PATH:", TEST_PATH)

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model folder not found: {MODEL_PATH}")

    if not os.path.isfile(TEST_PATH):
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

    id2label = model.config.id2label
    if isinstance(list(id2label.keys())[0], str):
        id2label = {int(k): v for k, v in id2label.items()}

    return tokenizer, model, id2label


def predict_seqc_edit_intent(model, tokenizer, id2label, text_src, text_tgt=""):
    input_text = build_input_text(text_src, text_tgt)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()

    return id2label[pred_id]


def evaluate_model(model, tokenizer, id2label):
    test_df = pd.read_csv(TEST_PATH)

    predictions = []
    true_labels = test_df["label"].tolist()
    label_list = list(id2label.values())

    for _, row in test_df.iterrows():
        pred = predict_seqc_edit_intent(model, tokenizer, id2label, row["text_src"], row["text_tgt"])
        predictions.append(pred)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted", zero_division=0
    )

    print("\nSEQC Test Results")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, labels=label_list, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=label_list)
    cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
    print(cm_df)


if __name__ == "__main__":
    tokenizer, model, id2label = load_model()
    print("\nSEQC model loaded successfully.\n")

    example_src = "He dont knows driving car."
    example_tgt = "He doesn't know how to drive a car."

    pred = predict_seqc_edit_intent(model, tokenizer, id2label, example_src, example_tgt)
    print("Sample Prediction:", pred)

    print("\nRunning full evaluation...\n")
    evaluate_model(model, tokenizer, id2label)