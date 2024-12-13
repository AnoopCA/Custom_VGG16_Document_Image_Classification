import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 9

# Define the VGG16 architecture
class VGG16(nn.Module):
    def __init__(self, num_classes=num_classes):
        super().__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        self.classifier = self._make_classifier()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _make_classifier(self):
        layers = []
        in_features = 512 * 7 * 7
        out_features = [4096, 4096, num_classes]
        for out_feat in out_features:
            layers += [nn.Linear(in_features, out_feat), nn.ReLU(inplace=True)]
            if out_feat != num_classes:
                layers += [nn.Dropout()]
            in_features = out_feat
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create an instance of the VGG16 model
vgg16 = VGG16().to(device)
print(vgg16)

# Define data transformations for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Define data transformations for testing (without random augmentations)
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Define path to the dataset
train_dataset = ImageFolder(root=r"D:\ML_Projects\Custom_VGG16_Document_Image_Classification\train", transform=train_transform)
test_dataset = ImageFolder(root=r"D:\ML_Projects\Custom_VGG16_Document_Image_Classification\test", transform=test_transform)
# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

# Training loop
def train(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%')

        model.eval()  # Set the model to evaluation mode
        test_correct = 0
        test_total = 0
        with torch.no_grad():  # Disable gradient computation during validation
            for test_inputs, test_labels in test_loader:
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_outputs = model(test_inputs)
                _, test_predicted = test_outputs.max(1)
                test_total += test_labels.size(0)
                test_correct += test_predicted.eq(test_labels).sum().item()
        test_accuracy = 100 * test_correct / test_total
        print(f'Testing - Epoch {epoch + 1}/{num_epochs}, Accuracy on test set: {test_accuracy:.2f}%')

# Define the number of training epochs
num_epochs = 10
# Train the model
train(vgg16, train_loader, test_loader, criterion, optimizer, num_epochs)
