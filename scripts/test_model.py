import torch
import torchvision.transforms as transforms
from PIL import Image
from model import RockPaperScissorsCNN

# Set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the model
model = RockPaperScissorsCNN()
model.load_state_dict(torch.load('/Applications/VSCode/MLProjs/rps/rock_paper_scissors_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

# Define class labels
class_labels = ['rock', 'paper', 'scissors']

# Define the preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust based on your model's input size
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize with mean=0.5 and std=0.5 for simplicity
])

def predict_image(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device

        # Get predictions
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label = class_labels[predicted.item()]

        print(f"The model predicts: {predicted_label}")
    except Exception as e:
        print(f"Error processing the image: {e}")

if __name__ == "__main__":
    # Replace with the path to your test image
    test_image_path = "/Applications/VSCode/MLProjs/rps/scripts/test_image.png"
    predict_image(test_image_path)