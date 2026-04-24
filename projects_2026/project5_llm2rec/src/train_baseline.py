import torch
from tqdm import tqdm

def train_model(model, train_loader, device, num_epochs=5):
    """
    Train the recommendation model.

    Uses CrossEntropy loss and Adam optimizer.

    Args:
        model: BERTSASRec model.
        train_loader: DataLoader for training data.
        device: CPU or GPU.
        num_epochs (int): Number of training epochs.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            item_seq = batch["item_seq"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            scores = model(item_seq, padding_mask)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")