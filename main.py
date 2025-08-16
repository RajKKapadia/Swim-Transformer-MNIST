import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# ==== 1. Data transforms ====
transform = transforms.Compose([
    transforms.Resize(224),  # Swin expects 224x224 by default
    transforms.Grayscale(num_output_channels=3),  # Convert 1 channel -> 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# ==== 2. Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=10)
model.to(device)

# ==== 3. Loss + Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)

# ==== 4. Training loop ====
for epoch in range(5):  # Small dataset, few epochs needed
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}")

# ==== 5. Evaluation ====
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

print(f"Test accuracy: {correct / total:.4f}")
