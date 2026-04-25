import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "gen_final_model")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "test.csv")

LABEL_LIST = ["Fact/Evidence", "Grammar", "Clarity", "Claim", "Other"]


def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_prompt(text_src, text_tgt=""):
    src = safe_text(text_src)
    tgt = safe_text(text_tgt)

    prompt = "Classify the edit intent into one of these labels: Fact/Evidence, Grammar, Clarity, Claim, Other.\n"
    if src and tgt:
        prompt += f"Source text: {src}\nEdited text: {tgt}\nAnswer:"
    elif src:
        prompt += f"Source text: {src}\nEdited text: \nAnswer:"
    else:
        prompt += f"Source text: \nEdited text: {tgt}\nAnswer:"
    return prompt


def clean_generated_text(text):
    text = str(text).strip()
    for label in LABEL_LIST:
        if text.lower() == label.lower():
            return label
    return "Other"


def load_model():
    print("MODEL_PATH:", MODEL_PATH)
    print("TEST_PATH:", TEST_PATH)

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"Model folder not found: {MODEL_PATH}")

    if not os.path.isfile(TEST_PATH):
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)

    return tokenizer, model


def predict_gen_edit_intent(model, tokenizer, text_src, text_tgt=""):
    prompt = build_prompt(text_src, text_tgt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=16
        )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_generated_text(pred)


def evaluate_model(model, tokenizer):
    test_df = pd.read_csv(TEST_PATH)

    predictions = []
    true_labels = test_df["label"].tolist()

    for _, row in test_df.iterrows():
        pred = predict_gen_edit_intent(model, tokenizer, row["text_src"], row["text_tgt"])
        predictions.append(pred)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted", zero_division=0
    )

    print("\nGEN Test Results")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, labels=LABEL_LIST, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=LABEL_LIST)
    cm_df = pd.DataFrame(cm, index=LABEL_LIST, columns=LABEL_LIST)
    print(cm_df)


if __name__ == "__main__":
    tokenizer, model = load_model()
    print("\nGEN model loaded successfully.\n")

    example_src = "He dont knows driving car."
    example_tgt = "He doesn't know how to drive a car."

    pred = predict_gen_edit_intent(model, tokenizer, example_src, example_tgt)
    print("Sample Prediction:", pred)

    print("\nRunning full evaluation...\n")
    evaluate_model(model, tokenizer)