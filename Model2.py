import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

# Configurare
dataset_dir = "synthetic_dataset"
model_save_path = "ocr_model_pytorch.pth"
img_size = (64, 64)
batch_size = 64
epochs = 30  # Număr de epoci pentru o eficacitate bună
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Obține lista de caractere din directoare
chars = sorted(os.listdir(dataset_dir))
num_classes = len(chars)
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}


# Dataset personalizat
class OCRDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Funcție pentru încărcarea datelor
def load_data():
    images = []
    labels = []

    print("Încărcare date...")
    # Parcurge toate directoarele de caractere
    for char in chars:
        char_dir = os.path.join(dataset_dir, char)
        image_files = glob.glob(os.path.join(char_dir, "*.png"))

        for img_path in image_files:
            img = Image.open(img_path).convert('L')  # Convertește la grayscale
            img = img.resize(img_size)
            img_array = np.array(img)

            images.append(img_array)
            labels.append(char_to_idx[char])

    # Convertim la numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Împărțim în date de antrenare și validare
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Transformări pentru augmentarea datelor
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Creăm dataset-urile
    train_dataset = OCRDataset(X_train, y_train, transform=train_transform)
    valid_dataset = OCRDataset(X_valid, y_valid, transform=valid_transform)

    # Creăm dataloader-ele
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, X_valid, y_valid


# Definirea arhitecturii CNN
class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()

        # Feature extraction - partea de convoluție
        self.features = nn.Sequential(
            # Primul bloc convoluțional
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Al doilea bloc convoluțional
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Al treilea bloc convoluțional
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Al patrulea bloc convoluțional
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Clasificare - partea fully connected
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Funcție pentru antrenament
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Antrenare"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass și optimizare
        loss.backward()
        optimizer.step()

        # Statistici
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# Funcție pentru validare
def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Validare"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistici
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# Funcție pentru vizualizarea rezultatelor
def plot_history(train_losses, valid_losses, train_accs, valid_accs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train')
    plt.plot(valid_accs, label='Validation')
    plt.title('Acuratețe')
    plt.xlabel('Epocă')
    plt.ylabel('Acuratețe')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epocă')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_pytorch.png')
    plt.show()


# Funcție pentru testarea modelului
def test_model(model, X_test, y_test, num_samples=10):
    # Transformare pentru testare
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model.eval()

    # Alege câteva imagini aleatorii
    indices = np.random.choice(range(len(X_test)), num_samples)
    test_images = X_test[indices]
    true_labels = y_test[indices]

    # Realizează predicții
    with torch.no_grad():
        predictions = []
        for img in test_images:
            img_tensor = test_transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())

    # Afișează imaginile și predicțiile
    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i], cmap='gray')
        plt.title(f"Pred: {idx_to_char[predictions[i]]}\nReal: {idx_to_char[true_labels[i]]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('test_examples_pytorch.png')
    plt.show()


# Funcția principală
def main():
    # Încarcă datele
    train_loader, valid_loader, X_valid, y_valid = load_data()
    print(
        f"Date încărcate: {len(train_loader.dataset)} imagini de antrenare, {len(valid_loader.dataset)} imagini de validare")

    # Inițializează modelul
    model = OCRModel(num_classes).to(device)
    print(f"Model creat și transferat pe {device}")

    # Criteriu și optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Listele pentru urmărirea progresului
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    # Cea mai bună acuratețe pentru salvarea modelului
    best_acc = 0.0

    # Buclă de antrenament
    print(f"Începe antrenamentul pentru {epochs} epoci...")
    for epoch in range(epochs):
        print(f"Epocă {epoch + 1}/{epochs}")

        # Antrenează o epocă
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

        # Actualizează scheduler-ul
        scheduler.step(valid_loss)

        # Salvează istoricul
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        # Afișează progresul
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

        # Salvează cel mai bun model
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Model salvat cu acuratețe {valid_acc:.4f}")

    # Încarcă cel mai bun model pentru testare
    model.load_state_dict(torch.load(model_save_path))

    # Vizualizează progresul
    plot_history(train_losses, valid_losses, train_accs, valid_accs)

    # Testează modelul
    test_model(model, X_valid, y_valid)

    print(f"Antrenament finalizat! Cel mai bun model salvat la {model_save_path}")
    print(f"Acuratețe maximă de validare: {best_acc:.4f}")


if __name__ == "__main__":
    main()