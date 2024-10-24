import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from model import SqueezeNet
import torch

# Data transformation pipeline
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and preprocess image
img = Image.open("/Users/micoria/Documents/python/DL/SqueezeNet/data/PALM-Validation400/V0008.jpg")
plt.imshow(img)  # Display the original image
img = data_transform(img)  # Apply transformations
img = torch.unsqueeze(img, dim=0)  # Add batch dimension

# Class names
name = ['非病理性近视', '病理性近视']

# Load model and weights
model_weight_path = r"/Users/micoria/Documents/python/DL/SqueezeNet/best_model.pth"
model = SqueezeNet(num_classes=4)  # Ensure the model has the right number of output classes
model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Inference
with torch.no_grad():
    output = torch.squeeze(model(img))  # Get model output

    predict = torch.softmax(output, dim=0)  # Apply softmax to get class probabilities
    predict_cla = torch.argmax(predict).item()  # Get index of the class with highest probability

    print(f'索引为: {predict_cla}')  # Output index
    print(f'预测结果为: {name[predict_cla]}, 置信度为: {predict[predict_cla].item()}')  # Output result and confidence

plt.show()  # Display the image
