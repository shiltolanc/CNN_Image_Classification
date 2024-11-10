import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torchvision.datasets as datasets
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
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = datasets.ImageFolder(root='G:/Other computers/My PC/University/2024_T2/COMP 309/Project/testdata', transform=transform)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # Load saved ensemble model weights and meta-model coefficients
    checkpoint = torch.load('model.pth', weights_only=False)

    # Initialize models and load their weights
    model_names = ['resnet50', 'vgg16', 'efficientnet_b0']
    models = [get_model(name, 3) for name in model_names]  # Adjust '3' if your classes count changes

    for name, model in zip(model_names, models):
        model.load_state_dict(checkpoint["models"][name])
        model.eval()

    # Reconstruct the meta-model using loaded coefficients and intercept
    meta_model = LogisticRegression()
    meta_model.coef_ = checkpoint["meta_model"]["coef"]
    meta_model.intercept_ = checkpoint["meta_model"]["intercept"]

    # Set the classes_ attribute
    num_classes = 3  # Adjust this as necessary
    meta_model.classes_ = np.arange(num_classes)

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
