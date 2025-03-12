# Classifying Real vs AI-Generated Images
<img src='https://media-hosting.imagekit.io//ef77d415f37542f1/Screenshot%202025-03-12%20at%201.10.11%20PM.png?Expires=1836411048&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=PfTp-hDs5R6bPffykgC5-IUiND7FNtH8U4Sbg00vjvSq9UB4NVFuGTLLmz6v0L8yjYiW5vcIzIZi7VZsl5JOgKdpg~xT5OTTrJqUBlZrQSCT~MiClF9T~24l~M5OXp~MOotf9mY9XfpQYuAycxESZ-NzuudLX0rYjUPI386DigoPggq2SmHcrhzGjYyAOyba3CKI8Er86TVjMt5wJBPVKze9jxQ2EaDruoZ-7hSO76kYMeRm6hGTRboc4zr9r9RpjXn-qEYwEsWORSmXS2AOJ1tv8VSPIaOMTJRyulrgrAIaeydjj7d6IOUzVYf~gB~zg60WIdi168Tenp3eW67KsQ__' width=242>
<img src='https://cdn.lucidpic.com/cdn-cgi/image/w=600,format=auto,metadata=none/66c4384702f8f.png' width=200>

With the rise of generative AI, realistic AI-generated human faces are increasingly indistinguishable from real ones, posing risks in identity fraud, misinformation, and digital security. Organizations need robust solutions to detect and differentiate between real and synthetic faces to prevent deepfake misuse. Current detection methods lack accuracy and interpretability, making them unreliable for high-stakes applications.

Our project addresses this challenge by developing an image classification framework using **Vision Transformers (ViTs) with Bayesian inference** for explainability. This model helps businesses, law enforcement, and media platforms detect AI-generated images with high confidence. By providing interpretable outputs, the framework enhances trust in automated detection systems.

This solution is crucial for identity verification, content moderation, and safeguarding digital assets against AI-driven manipulation. As synthetic media continues to evolve, having a reliable and transparent classification system is essential for mitigating risks.

# Data

<a href='https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset'>Kaggle dataset</a> with over 4,500 real and AI-generated images each.


# Process

## Required Packages
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
```
**PyTorch** serves as the core deep learning framework, providing tensor operations (`torch`), neural network layers (`torch.nn`), and optimization algorithms (`torch.optim`). For handling image data efficiently, **torchvision** is used, including `transforms` for preprocessing (e.g., resizing, normalization) and `datasets` for loading real and AI-generated face datasets. To enhance model performance, we utilize **Hugging Face's Transformers**, specifically `AutoFeatureExtractor` for automated image feature extraction and `AutoModel` to load a pre-trained **Vision Transformer (ViT)** for classification. 

## Load and Preprocess Data
```
transform = transforms.Compose([
transforms.Resize((128, 128)), # Reduce resolution to lower computational cost
transforms.ToTensor(),
])

dataset_path = 'Human Faces Dataset'
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```
We load and preprocess the dataset to ensure efficient training while minimizing computational overhead. We use **torchvision.transforms** to resize images to **128x128 pixels**, reducing memory consumption and speeding up model training. The images are then converted into tensors using `ToTensor()`, making them compatible with PyTorch models.

The dataset, stored in **ImageFolder format**, is loaded from the specified path and categorized into two classes: **'AI-Generated Images'** and **'Real Images'**. We use a **batch size of 16** in the `DataLoader` to balance computational efficiency and memory usage, while shuffling the data ensures randomness during training.

## Load Pre-Trained Vision Transformer (ViT) and Extract Features
```
dino_model_name = "facebook/dino-vits16"
feature_extractor = AutoFeatureExtractor.from_pretrained(dino_model_name)
dino_model = AutoModel.from_pretrained(dino_model_name)
dino_model.eval()

def  extract_features(images):
	with torch.no_grad():
	inputs = feature_extractor(images, return_tensors="pt")
	outputs = dino_model(**inputs)

	return outputs.last_hidden_state.mean(dim=1) # Global average pooling
```
We use a **pre-trained Vision Transformer (ViT) model** from Meta's **DINO (Self-Supervised Learning)** to extract meaningful image features. We load the `facebook/dino-vits16` model, and the `AutoFeatureExtractor` preprocesses input images into the format required by the model.

The `extract_features` function takes a batch of images, processes them through the feature extractor, and passes them to the DINO model. We apply **global average pooling** (`mean(dim=1)`) to obtain a compact feature representation for each image, which will be used as input for classification.

## Train-Test Split and DataLoader Creation
```
from torch.utils.data import random_split, DataLoader
  
train_size = int(0.8 * len(dataset)) # 80% Train, 20% Test
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train Size: {len(train_dataset)}  | Test Size: {len(test_dataset)}")
```
To train and evaluate the classification model, we split the dataset into **training (80%)** and **testing (20%)** sets. This ensures that the model is trained on a diverse set of images while maintaining a separate evaluation set for performance assessment.

We use `random_split()` from **torch.utils.data** to randomly divide the dataset into **train** and **test** subsets. Each subset is then wrapped in a `DataLoader`, which efficiently loads data in batches, reducing memory usage and improving training speed.

## Bayesian Neural Network for Classification
```
class  BayesianSimpleCNN(nn.Module):
	def  __init__(self, num_classes, dropout_rate=0.5):
		super(BayesianSimpleCNN, self).__init__()
		self.fc1 = nn.Linear(384, 128)
		self.dropout1 = nn.Dropout(dropout_rate) # Monte Carlo Dropout
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(128, 64)
		self.dropout2 = nn.Dropout(dropout_rate)
		self.fc3 = nn.Linear(64, num_classes) # Final classification layer

	def  forward(self, x):
		x = self.dropout1(x) # Apply dropout before first FC layer
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout2(x) # Apply dropout before second FC layer
		x = self.fc2(x)
		x = self.fc3(x) # Final output layer
		return x

num_classes = len(dataset.classes)
classifier = BayesianSimpleCNN(num_classes, dropout_rate=0.3)
```
We implement a **Bayesian-inspired Simple CNN classifier** using fully connected layers. Instead of a traditional deterministic neural network, we integrate **Monte Carlo Dropout**, which simulates Bayesian inference by introducing uncertainty estimation.

 - Input Layer: Takes **384-dimensional features** extracted from the DINO ViT model.  
 
 - Fully Connected Layers (FCs):
  --   FC1 (128 units): Learns intermediate representations.
  --   FC2 (64 units): Further refines features before classification.
  --   FC3 (Output Layer): Maps features to `num_classes` (2 in this case: **Real** vs **AI-Generated**).  
  - **Monte Carlo Dropout (`Dropout`):**
  --   Applied before `FC1` and `FC2` layers to induce uncertainty in predictions.
  --   Helps estimate confidence in classifications during inference.  
 - Activation Function: **ReLU** is used to introduce non-linearity for better feature learning.


## Model Training
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

def  train_model():
	classifier.train()
	for epoch in  range(30):
		running_loss = 0.0

		for images, labels in train_loader:
			features = extract_features(images) # Get DINOv2 features
			optimizer.zero_grad()
			outputs = classifier(features)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

			print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

train_model()
```
**Loss Function**: `CrossEntropyLoss()` is suitable for binary and multi-class classification problems.
**Optimizer** – `Adam` with Learning Rate = 0.001. It adjusts learning rates dynamically for each parameter, making training more stable and efficient.

**Model training**: `train_model()` 

Iterate Over Mini-Batches from `train_loader`:
-   Extract **DINO features** for each batch using `extract_features(images)`.
-   Compute model predictions.
-   Compute **loss** using `CrossEntropyLoss()`.
-   Perform **backpropagation** (`loss.backward()`) and update weights using `optimizer.step()`.  
-   Print **Loss per Epoch**

## Model Evaluation
```
def  evaluate_model():
	classifier.eval() # Set the model to evaluation mode
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in test_loader:
			features = extract_features(images)
			outputs = classifier(features)

			_, predicted = torch.max(outputs, 1) # Get class with highest probability
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	accuracy = 100 * correct / total
	print(f"Test Accuracy: {accuracy:.2f}%")

evaluate_model()
```
- **Extract DINO Features**: Images are transformed into feature vectors using `extract_features(images)`.  

- **Compute Model Predictions**:
 --   `outputs = classifier(features)` produces raw logits
 --   `torch.max(outputs, 1)` selects the class with the highest probability
- **Compute Accuracy**: The number of correct predictions vs. total samples

## Bayesian Inference for Uncertainty Estimation
Now that the model is trained, we implement **Bayesian Inference using Monte Carlo Dropout** to estimate model confidence levels for each prediction. Instead of making a single deterministic prediction, we perform multiple forward passes (num_samples = 200) while keeping dropout layers active.

While Step 6 gives us accuracy, it doesn't provide insights into **how confident the model is** in its predictions. This step adds uncertainty estimation, making our classifier more robust, especially for ambiguous images.
```
def  bayesian_inference(model, feature_vector, num_samples=200):
	model.train() # Keep dropout active during inference
	preds = []
	with torch.no_grad():
		for i in  range(num_samples):
			output = model(feature_vector)
			probs = torch.nn.functional.softmax(output, dim=1) # Convert logits to probabilities
			preds.append(probs.cpu().numpy())

			if i < 5:
			print(f"Iteration {i+1}: {probs.cpu().numpy()}")

	preds = np.array(preds)
	mean_pred = preds.mean(axis=0) # Average probability across runs
	uncertainty = preds.std(axis=0) # Standard deviation (uncertainty measure)

	return mean_pred, uncertainty
```

- **`model.train()` During Inference** – Unlike normal inference, we keep dropout enabled to introduce randomness in predictions

- **Perform Multiple Stochastic Passes** (`num_samples = 200`) – Each pass provides a slightly different output due to the stochastic dropout.
- **Convert Logits to Probabilities** – Use `softmax` to get class probabilities.  
- **Compute Mean Prediction** – The average probability over multiple runs gives a more stable prediction.
- **Estimate Uncertainty** – The standard deviation of predictions provides an uncertainty measure. Higher variance means the model is less confident.

**Why This is Useful?**
-   **Trust & Reliability** – Helps detect cases where the model is uncertain.
-   **Better Decision-Making** – If uncertainty is high, human review may be required.
-   **Adversarial & Edge Case Detection** – Can identify hard-to-classify images.


## Feature Importance Analysis Using Gradients
Determining which features contributed the most to a classification decision.
```
def  compute_feature_importance(model, feature_vector):
	model.train() # Set model to train mode
	feature_vector = feature_vector.clone().requires_grad_(True) # Enable gradient computation

	# Forward pass
	output = model(feature_vector)
	predicted_class = torch.argmax(output, dim=1) # Get the predicted class

	# Backward pass for the predicted class
	output[:, predicted_class].sum().backward() # Sum to create a scalar for backward pass

	# Compute feature importance as the absolute gradient values
	feature_importance = torch.abs(feature_vector.grad).mean(dim=0).cpu().numpy()
	return feature_importance
```
- **Forward Pass Through Model**: Computes the output logits for classification.  

- **Identify Predicted Class**: Using `torch.argmax(output, dim=1)`, we select the most probable class.  
- **Backward Pass for Gradient Computation**:
 --   We call `.backward()` on the predicted class’s output. This propagates gradients back to the feature vector, showing which parts of the feature representation were most influential.
 -- **Compute Feature Importance**: The absolute values of the **gradients** tell us how much each feature contributed to the prediction.
