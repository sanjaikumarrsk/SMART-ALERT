import argparse
import os
import time
from tqdm import tqdm

import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

from utils import accuracy, save_checkpoint


def get_dataloaders(dataset_dir, batch_size, num_workers=4, image_size=224):
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes=2, feature_extract=True, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs, labels) * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def validate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_acc += accuracy(outputs, labels) * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train a binary detector using ResNet18 transfer learning')
    parser.add_argument('--dataset-dir', required=True, help='Path to dataset folder containing train/val subfolders')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', required=True, help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--fine-tune', action='store_true', help='Unfreeze full model for fine-tuning')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience on validation loss')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and (not args.no_cuda) else 'cpu')

    train_loader, val_loader, classes = get_dataloaders(args.dataset_dir, args.batch_size, args.num_workers, args.image_size)

    model = build_model(num_classes=len(classes), feature_extract=(not args.fine_tune), pretrained=True)
    model = model.to(device)

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

    best_val_loss = float('inf')
    best_epoch = -1
    start_time = time.time()
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)

        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss}, is_best, args.output_dir)

        print(f'  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}')
        print(f'  Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.4f}')
        print(f'  Best Val Loss: {best_val_loss:.4f} (epoch {best_epoch})')

        if early_stop_counter >= args.patience:
            print('Early stopping triggered')
            break

    elapsed = time.time() - start_time
    print(f'Training complete in {elapsed/60:.2f} minutes. Best val loss: {best_val_loss:.4f} (epoch {best_epoch})')


if __name__ == '__main__':
    main()
