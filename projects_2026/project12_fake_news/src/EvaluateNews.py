#Ahmed Mohamed
#DS8008 Final Project

class EvaluateNews:
    def __init__(self):
        import torch
        import torch.nn as nn
        from transformers import BertTokenizer, BertModel

        self.torch = torch
        self.nn = nn
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel.from_pretrained("bert-base-uncased")
                self.fc = nn.Linear(768, 2)

            def forward(self, input_ids, attention_mask):
                output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls = output.last_hidden_state[:, 0, :]
                return self.fc(cls)

        self.model = Model().to(self.device)

    def train(self, train_path, epochs=3):
        import os

        texts = []
        labels = []

        for folder, label in [("real", 0), ("fake", 1)]:
            folder_path = os.path.join(train_path, folder)
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                        texts.append(f.read())
                        labels.append(label)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        labels = self.torch.tensor(labels).to(self.device)

        loss_fn = self.nn.CrossEntropyLoss()
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=2e-5)

        for _ in range(epochs):
            self.model.train()
            logits = self.model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict_folder(self, test_path):
        import os

        results = {}
        self.model.eval()

        for file_name in os.listdir(test_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(test_path, file_name), "r", encoding="utf-8") as f:
                    text = f.read()

                encoded = self.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                with self.torch.no_grad():
                    logits = self.model(input_ids, attention_mask)
                    probs = self.torch.softmax(logits, dim=1)[0]

                results[file_name] = {
                    "real_prob": probs[0].item(),
                    "fake_prob": probs[1].item(),
                    "prediction": "FAKE" if probs[1] > probs[0] else "REAL"
                }

        return results
