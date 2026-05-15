import torch
from torchvision import transforms
from PIL import Image, ImageOps
import gradio as gr

from model import DigitCNN


MODEL_PATH = "models/mnist_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DigitCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def create_blank_canvas():
    return {
        "background": Image.new("RGB", (280, 280), "white"),
        "layers": [],
        "composite": Image.new("RGB", (280, 280), "white")
    }


def preprocess_image(editor_value):
    if editor_value is None:
        raise gr.Error("Please draw a digit first.")

    image = editor_value.get("composite")

    if image is None:
        raise gr.Error("Please draw a digit first.")

    # Convert transparent/colored image to grayscale
    image = image.convert("RGBA")
    white_background = Image.new("RGBA", image.size, "white")
    white_background.alpha_composite(image)

    grayscale = white_background.convert("L")

    # User draws black on white, but MNIST is white digit on black background
    inverted = ImageOps.invert(grayscale)

    # Crop around the drawn digit
    bbox = inverted.point(lambda p: 255 if p > 30 else 0).getbbox()

    if bbox is None:
        raise gr.Error("Please draw a digit first.")

    cropped = inverted.crop(bbox)

    # Put the digit on a square canvas
    width, height = cropped.size
    square_size = max(width, height)

    square = Image.new("L", (square_size, square_size), 0)
    paste_x = (square_size - width) // 2
    paste_y = (square_size - height) // 2
    square.paste(cropped, (paste_x, paste_y))

    # Add padding so the digit is not touching the edges
    padding = int(square_size * 0.25)
    padded = Image.new("L", (square_size + padding * 2, square_size + padding * 2), 0)
    padded.paste(square, (padding, padding))

    # Resize to MNIST size: 28x28
    resized = padded.resize((28, 28), Image.Resampling.LANCZOS)

    tensor = transform(resized).unsqueeze(0).to(device)

    return tensor, resized


def predict_digit(editor_value):
    image_tensor, processed_image = preprocess_image(editor_value)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_digit = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_digit].item()

    confidence_scores = {
        str(i): float(probabilities[i]) for i in range(10)
    }

    prediction_text = f"Predicted digit: {predicted_digit} ({confidence * 100:.2f}% confidence)"

    return prediction_text, confidence_scores, processed_image


demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.ImageEditor(
        value=create_blank_canvas(),
        type="pil",
        label="Draw a digit here",
        height=350,
        width=350
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Confidence Scores", num_top_classes=3),
        gr.Image(label="What the model sees", type="pil")
    ],
    title="Handwritten Digit Recognition App",
    description=(
        "Draw a digit from 0 to 9 using a dark brush on the white canvas. "
        "The PyTorch CNN model will predict the digit and show its confidence score."
    )
)


if __name__ == "__main__":
    demo.launch()