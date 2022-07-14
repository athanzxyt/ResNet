import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from resnet import ResNet18
import matplotlib.pyplot as plt

# Define transformations
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5], inplace=True)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5], inplace=True)
])

# Create datasets
dataset = torchvision.datasets.CIFAR10(root="data/", download=True, transform=train_transform)
val_ratio = 0.1
train_dataset, val_dataset = random_split(dataset, [int((1-val_ratio) * len(dataset)), int(val_ratio * len(dataset))])
test_dataset = torchvision.datasets.CIFAR10(root="data/", download=True, train=False, transform=test_transform)

# Create dataloaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Find device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Network parameters
model = ResNet18(3,10)
model.to(device)
epochs = 40
lr = 1e-1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*epochs),int(0.75*epochs)], gamma=0.1)

# Train the network
train_acc = []
val_acc = []

for epoch in range(epochs):
    # Train
    print(f'Epoch: {epoch}')
    model.train()
    train_loss, correct, total = 0,0,0
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predictions = outputs.max(1)
        total += len(labels)
        correct += predictions.eq(labels).sum().item()

        print(f'Iter: {idx}\t\tLoss: {round(train_loss/(idx+1),5)}\t\tAcc: {round(correct/total,5)}')
        train_acc.append(correct/total)

    # Validate
    model.eval()
    val_loss, correct, total = 0,0,0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Accuracy
            outputs = model(images)
            _, predictions = outputs.max(1)
            total += len(labels)
            correct += predictions.eq(labels).sum().item()

        val_acc.append(correct/total)

    # Iterate
    scheduler.step()

# Test
model.eval()
test_loss, correct_top1, correct_top5, total = 0,0,0,0
with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        if idx > 0: continue
        images, labels = images.to(device), labels.to(device)
        
        # Accuracy
        outputs = model(images)
        _, predictions = outputs.max(1)
        total += len(labels)
        correct_top1 += predictions.eq(labels).sum().item()

        has5 = torch.topk(outputs,5).indices.eq(labels.unsqueeze(1).repeat(1, 5))
        correct_top5 = torch.any(has5, 1).sum().item()

print(f'TOP1 ACC: {correct_top1/total}')
print(f'TOP5 ACC: {correct_top5/total}')

plt.plot(range(len(train_acc)), train_acc, label='train', color='red')
plt.plot([(len(train_loader)) * i for i in range(len(val_acc))], val_acc, label='val', color='blue')
plt.legend()
plt.savefig('acc.png')

 

