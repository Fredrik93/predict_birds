import torch
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights

from PIL import Image

bird_keywords = [
    "bird", "robin", "eagle", "sparrow", "parrot", "owl",
    "hawk", "falcon", "penguin", "flamingo",
    "woodpecker", "duck", "goose", "heron"
]


# 1. Load a pre-trained model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()  # very important

# 2. Define image transformations (what the model expects)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Load image
image = Image.open("images/bald_eagle.jpg").convert("RGB")
image = transform(image)
image = image.unsqueeze(0)  # add batch dimension

# 4. Make prediction
with torch.no_grad():
    outputs = model(image)

# 5. Get predicted class
_, predicted = outputs.max(1)


with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]


probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

best_bird = None
best_prob = 0.0

for i, label in enumerate(labels):
    if any(word in label.lower() for word in bird_keywords):
        prob = probabilities[i].item()
        if prob > best_prob:
            best_prob = prob
            best_bird = label

if best_bird:
    print(f"Predicted bird: {best_bird} ({best_prob:.2%})")
else:
    print("No bird detected")