import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import DigitCNN


def predict_sample():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    image, true_label = test_dataset[0]

    model = DigitCNN().to(device)
    model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
    model.eval()

    image_for_model = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_for_model)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_label = torch.max(probabilities, 1)

    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label.item()}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Prediction: {predicted_label.item()} | True: {true_label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    predict_sample()