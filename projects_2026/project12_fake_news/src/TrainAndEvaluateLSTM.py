#divya bharathi
def TrainAndEvaluateLSTM(AllArticles, epochs=15):
    import torch
    import torch.nn as nn
    import os
    import random
    from collections import Counter
    from torch.utils.data import DataLoader, TensorDataset

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

    print(f"Train size: {len(train_texts)}")
    print(f"Test size: {len(test_texts)}")

    def tokenize(text):
        return text.lower().split()

    tokenized_train = [tokenize(t) for t in train_texts]

    counter = Counter()
    for tokens in tokenized_train:
        counter.update(tokens)

    vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(20000))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    def encode(text, max_len=300):
        tokens = tokenize(text)
        ids = [vocab.get(w, 1) for w in tokens]
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        return ids

    X_train = torch.tensor([encode(t) for t in train_texts])
    X_test = torch.tensor([encode(t) for t in test_texts])

    y_train = torch.tensor(train_labels)
    y_test = torch.tensor(test_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    # MODEL
    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
            self.lstm = nn.LSTM(128, 128, batch_first=True)
            self.fc = nn.Linear(128, 2)

        def forward(self, x):
            x = self.embedding(x)
            _, (hidden, _) = self.lstm(x)
            return self.fc(hidden[-1])

    model = LSTMClassifier(len(vocab)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 32

    # TRAINING
    #outputs = model(X_train)
    #loss = loss_fn(outputs, y_train)
    #Above 2 lines are too heavy, can't load all the articles at once or we just run out of memory have to batch it
    TrainDataset  = TensorDataset(X_train,y_train)
    BatchSize = 128
    TrainLoader = DataLoader(TrainDataset,batch_size=BatchSize,shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for BatchX, BatchY in TrainLoader:
            BatchX = BatchX.to(device)
            BatchY = BatchY.to(device)

            optimizer.zero_grad()

            outputs = model(BatchX) 
            loss = loss_fn(outputs, BatchY)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # EVALUATION, similarily adding batching here
    TestDataset = TensorDataset(X_test,y_test)
    TestLoader = DataLoader(TestDataset,batch_size=128)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for BatchX, BatchY in TestLoader:
            BatchX = BatchX.to(device)
            BatchY = BatchY.to(device)

            outputs = model(BatchX)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == BatchY).sum().item()
            total += BatchY.size(0)

    accuracy = correct/total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy