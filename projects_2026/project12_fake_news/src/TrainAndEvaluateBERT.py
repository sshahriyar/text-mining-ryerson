#divya bharathi
def TrainAndEvaluateBERT(AllArticles, epochs=3):
    import torch
    import torch.nn as nn
    import os
    import random
    from collections import Counter
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import BertTokenizer, BertModel

    random.shuffle(AllArticles)

    train_texts = [row[0] for row in AllArticles]
    train_labels = [row[1] for row in AllArticles]

    AllTestArticles = []
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # project/
    RealTestPath = os.path.join(
        PROJECT_ROOT,
        "Data",
        "test",
        "RealNewsArticlesTestingSet"
    )
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # project/
    FakeTestPath = os.path.join(
        PROJECT_ROOT,
        "Data",
        "test",
        "FakeNewsArticlesTestingSet"
    )

    # Real = 0
    for file in os.listdir(RealTestPath):
        with open(os.path.join(RealTestPath, file), "r", encoding="utf-8", errors="ignore") as f:
            AllTestArticles.append((f.read(), 0))

    # Fake = 1
    for file in os.listdir(FakeTestPath):
        with open(os.path.join(FakeTestPath, file), "r", encoding="utf-8", errors="ignore") as f:
            AllTestArticles.append((f.read(), 1))

    test_texts = [row[0] for row in AllTestArticles]
    test_labels = [row[1] for row in AllTestArticles]

    #print(f"Train size: {len(train_texts)}")
    #print(f"Test size: {len(test_texts)}")
    #Using pretrained BERT for tokenization
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(text):
        return text.lower().split()

    tokenized_train = [tokenize(t) for t in train_texts]
    def encode_batch(texts, max_len=256):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

    train_enc = encode_batch(train_texts)
    test_enc = encode_batch(test_texts)

    X_train_ids = train_enc["input_ids"]
    X_train_mask = train_enc["attention_mask"]

    X_test_ids = test_enc["input_ids"]
    X_test_mask = test_enc["attention_mask"]

    y_train = torch.tensor(train_labels)
    y_test = torch.tensor(test_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class BERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.fc = nn.Linear(768, 2)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
            return self.fc(cls_output)

    model = BERTClassifier().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 32

    TrainDataset = TensorDataset(X_train_ids, X_train_mask, y_train)
    TrainLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)

    TestDataset = TensorDataset(X_test_ids, X_test_mask, y_test)
    TestLoader = DataLoader(TestDataset, batch_size=batch_size)

    # TRAINING
    #outputs = model(X_train)
    #loss = loss_fn(outputs, y_train)
    #Above 2 lines are too heavy, can't load all the articles at once or we just run out of memory have to batch it
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, (BatchIDs, BatchMask, BatchY) in enumerate(TrainLoader):
            BatchIDs = BatchIDs.to(device)
            BatchMask = BatchMask.to(device)
            BatchY = BatchY.to(device)

            optimizer.zero_grad()

            outputs = model(BatchIDs, BatchMask)
            loss = loss_fn(outputs, BatchY)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:  # print every 10 batches
                print(f"Epoch {epoch+1} | Batch {i}/{len(TrainLoader)} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} DONE | Total Loss: {total_loss:.4f}")

    # EVALUATION, similarily adding batching here
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for BatchIDs, BatchMask, BatchY in TestLoader:
            BatchIDs = BatchIDs.to(device)
            BatchMask = BatchMask.to(device)
            BatchY = BatchY.to(device)

            outputs = model(BatchIDs, BatchMask)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == BatchY).sum().item()
            total += BatchY.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy