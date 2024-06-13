import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from DenseNet import DenseNet

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

BATCH_SIZE = 32
EPOCHS = 30
K_FOLDS = 5

transformations = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='../DenseNetTrainingSetKFold',
                                transform=transformations)

kfold = KFold(n_splits=K_FOLDS, shuffle=True)

def train(model, train_loader, optimizer, criterion, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: [{batch_idx * len(image)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tTrain Loss: {loss.item():.6f}")

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    print(f'Fold {fold + 1}/{K_FOLDS}')
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

    model = DenseNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    fold_epoch_accuracies = []
    
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}')
        train(model, train_loader, optimizer, criterion, log_interval=200)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        fold_epoch_accuracies.append(test_accuracy)
        print(f"\nFold {fold + 1}, Epoch {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n")
    
    fold_accuracies.append(fold_epoch_accuracies)

# Plot test accuracy for each epoch per fold
plt.figure()
for fold in range(K_FOLDS):
    plt.plot(range(1, EPOCHS + 1), fold_accuracies[fold], marker='o', label=f'Fold {fold + 1}')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs Epoch per Fold')
plt.legend()
plt.grid()
plt.show()
