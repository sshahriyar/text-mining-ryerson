import os
import torch
import torch.nn as nn
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from sklearn.metrics import classification_report, accuracy_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "model", "snet_final_model")
DATA_DIR = os.path.join(BASE_DIR, "data")

TEST_FILE = os.path.join(DATA_DIR, "test.csv")

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
labels = sorted(train_df["label"].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

print("Label Mapping:", label2id)



# SNET MODEL 

class SNet(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  

        # EXACT MATCH WITH YOUR CHECKPOINT
        self.fc1 = nn.Linear(hidden * 3, hidden)  
        self.norm1 = nn.LayerNorm(hidden)

        self.fc2 = nn.Linear(hidden, 384)          
        self.norm2 = nn.LayerNorm(384)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(384, num_labels)

        self.act = nn.ReLU()

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state
        batch_size = input_ids.size(0)

        # RoBERTa </s> token id
        sep_token_id = 2
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=False)

        o_list = []
        n_list = []

        for i in range(batch_size):
            positions = sep_positions[sep_positions[:, 0] == i][:, 1]

            if len(positions) < 2:
                raise ValueError("Separator tokens not found")

            src_end = positions[0]
            tgt_end = positions[1]

            o = hidden_states[i, src_end - 1, :]
            n = hidden_states[i, tgt_end - 1, :]

            o_list.append(o)
            n_list.append(n)

        o = torch.stack(o_list)
        n = torch.stack(n_list)
        interaction = torch.cat([o, n, o - n], dim=1)  # 2304

        x = self.fc1(interaction)
        x = self.norm1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.dropout(x)

        logits = self.classifier(x)
        return logits



# LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = SNet(MODEL_NAME, len(labels))
state_dict = load_file(os.path.join(MODEL_DIR, "model.safetensors"))

model.load_state_dict(state_dict)  
model.to(device)
model.eval()

print("SNET Model Loaded Successfully")



def predict(src_text, tgt_text):
    enc = tokenizer(
        str(src_text),
        str(tgt_text),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc)
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    return pred_id



# TEST EVALUATION
test_df = pd.read_csv(TEST_FILE)
y_true = []
y_pred = []
for _, row in test_df.iterrows():
    src = row["text_src"]
    tgt = row["text_tgt"]

    pred = predict(src, tgt)

    y_pred.append(pred)
    y_true.append(label2id[row["label"]])



# METRICS
print("\nRESULTS")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(labels))]
    )
)



print("\nPREDICTION")
example_src = "The experiment shows better results in recent trials."
example_tgt = "The experiment shows results."
pred_id = predict(example_src, example_tgt)
print("Prediction:", id2label[pred_id])