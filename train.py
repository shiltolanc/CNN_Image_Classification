# %%
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights, VGG16_Weights, EfficientNet_B0_Weights

# Initialize the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define function to load and modify pre-trained model
def get_model(model_name, num_classes):
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")

    return model.to(device)

def main():

    # Data transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset
    batch_size = 4
    trainset = torchvision.datasets.ImageFolder(
        root='G:/Other computers/My PC/University/2024_T2/COMP 309/Project/trainingdata', transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder(
        root='G:/Other computers/My PC/University/2024_T2/COMP 309/Project/testdata', transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('cherry', 'strawberry', 'tomato')


    # List of models to use in the ensemble
    model_names = ['resnet50', 'vgg16', 'efficientnet_b0']
    models = [get_model(name, len(classes)) for name in model_names]

    # Fine-tune each model individually
    num_epochs = 3  # Set lower for faster training
    learning_rate = 0.001
    meta_features, meta_labels = [], []

    # Prepare to collect losses and accuracies
    losses = []
    accuracies = []

    # Training each model
    for model in models:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        model_losses = []
        model_accuracies = []

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                running_loss += loss.item()

                # Accuracy calculation
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                loss.backward()
                optimizer.step()

            # Record loss and accuracy for this epoch
            model_losses.append(running_loss / len(trainloader))
            epoch_accuracy = correct_predictions / total_predictions
            model_accuracies.append(epoch_accuracy)

            print(f"Model {model.__class__.__name__}, Epoch {epoch + 1}/{num_epochs}, "
                  f"Loss: {running_loss / len(trainloader)}, Accuracy: {epoch_accuracy * 100:.2f}%")

        # Append model losses and accuracies to main lists
        losses.append(model_losses)
        accuracies.append(model_accuracies)

    # Collect meta-features for stacking
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            predictions = [model(inputs).cpu().numpy() for model in models]
            meta_features.extend(np.hstack(predictions))
            meta_labels.extend(labels.cpu().numpy())

    meta_features = np.array(meta_features)
    meta_labels = np.array(meta_labels)

    # Split meta-features for training the meta-model
    X_train, X_val, y_train, y_val = train_test_split(meta_features, meta_labels, test_size=0.2, random_state=42)

    # Train a meta-model (Logistic Regression)
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train, y_train)

    # Evaluate the meta-model
    y_pred = meta_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Ensemble Model Validation Accuracy: {accuracy * 100:.2f}%")

    # Save model weights and meta-model coefficients to .pth file
    save_dict = {
        "models": {name: model.state_dict() for name, model in zip(model_names, models)},
        "meta_model": {
            "coef": meta_model.coef_,
            "intercept": meta_model.intercept_
        }
    }

    torch.save(save_dict, 'model.pth')

    # Test the final ensemble model
    with torch.no_grad():
        meta_test_features, meta_test_labels = [], []
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            predictions = [model(inputs).cpu().numpy() for model in models]
            meta_test_features.extend(np.hstack(predictions))
            meta_test_labels.extend(labels.cpu().numpy())

    meta_test_features = np.array(meta_test_features)
    meta_test_labels = np.array(meta_test_labels)
    test_predictions = meta_model.predict(meta_test_features)
    test_accuracy = accuracy_score(meta_test_labels, test_predictions)
    print(f"Ensemble Model Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
