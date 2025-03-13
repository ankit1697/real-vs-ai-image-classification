# Classifying Real vs AI-Generated Images
<p float="left">
<img src='https://media-hosting.imagekit.io//ef77d415f37542f1/Screenshot%202025-03-12%20at%201.10.11%20PM.png?Expires=1836411048&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=PfTp-hDs5R6bPffykgC5-IUiND7FNtH8U4Sbg00vjvSq9UB4NVFuGTLLmz6v0L8yjYiW5vcIzIZi7VZsl5JOgKdpg~xT5OTTrJqUBlZrQSCT~MiClF9T~24l~M5OXp~MOotf9mY9XfpQYuAycxESZ-NzuudLX0rYjUPI386DigoPggq2SmHcrhzGjYyAOyba3CKI8Er86TVjMt5wJBPVKze9jxQ2EaDruoZ-7hSO76kYMeRm6hGTRboc4zr9r9RpjXn-qEYwEsWORSmXS2AOJ1tv8VSPIaOMTJRyulrgrAIaeydjj7d6IOUzVYf~gB~zg60WIdi168Tenp3eW67KsQ__' width=300>
<img src='https://media-hosting.imagekit.io//f4f04f412470469c/100k-ai-faces-6.jpg?Expires=1836494263&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=BWdWHOItUOfLzylPQ90Grdrf5xUssYxu7b69sYCUosnUsuvAWL~0V2I4PFLppmigd7YcCxJWH5W7zlbJk8tZz7Q9UDXy9xpk5pWoTWLOxRoUVoAgcikAbIa7r4Q6BnxnjETLJ1Bik6zh-54E-u3H9Zmj0MHPWc8E~3Pwk3ebfcIR9iOMuPFbMYGsVBJFTPpNpi5yYWzX0s33waYppnbPu-LES1mMTB-uz-hByJNjnQz2JEeC0d9hYb5fKURYK3PjzjPZ1coUOetAFIhpwni5q3s2WYILrOyxXClMgUO~irST68fkdX10B~wvUGkhyngSki591lhBQxoPwsiTCg9OjQ__' width=300>
</p>


# Table of Contents
1. [Problem Statement](#problem-statement)
2. [The Difference](#the-difference)
3. [Data](#data)
4. [Process](#process)<br>
	3.1. [Required Packages](#required-packages)<br>
	3.2. [Load and Preprocess Data](#load-and-preprocess-data)<br>
	3.3. [Load Pre-Trained Vision Transformer (ViT) and Extract Features](#load-pre-trained-vision-transformer-vit-and-extract-features)<br>
	3.4. [Train-Test Split and DataLoader Creation](#train-test-split-and-dataloader-creation)<br>
	3.5. [Bayesian Neural Network for Classification](#bayesian-neural-network-for-classification)<br>
	3.6. [Model Training](#model-training)<br>
	3.7. [Model Evaluation](#model-evaluation)<br>
	3.8. [Bayesian Inference for Uncertainty Estimation](#bayesian-inference-for-uncertainty-estimation)<br>
	3.9. [Feature Importance Analysis Using Gradients](#feature-importance-analysis-using-gradients)<br>
	3.10. [Visualizing Feature Importance on the Image](#visualizing-feature-importance-on-the-image)<br>
	3.11. [Occlusion Sensitivity](#occlusion-sensitivity)
5. [Classifying real-world images](#classifying-real-world-images)<br>
4.1. [Final Output](#final-output)


# Problem Statement

With the rise of generative AI, realistic AI-generated human faces are increasingly indistinguishable from real ones, posing risks in identity fraud, misinformation, and digital security. Organizations need robust solutions to detect and differentiate between real and synthetic faces to prevent deepfake misuse. Current detection methods lack accuracy and interpretability, making them unreliable for high-stakes applications.

Our project addresses this challenge by developing an image classification framework using **Vision Transformers (ViTs) with Bayesian inference** for explainability. This model helps businesses, law enforcement, and media platforms detect AI-generated images with high confidence. By providing interpretable outputs, the framework enhances trust in automated detection systems.

This solution is crucial for identity verification, content moderation, and safeguarding digital assets against AI-driven manipulation. As synthetic media continues to evolve, having a reliable and transparent classification system is essential for mitigating risks.

# The Difference
What makes AI-generated images different from Real images

<img src='https://media-hosting.imagekit.io//6f352af2b44b40f4/DALL%C2%B7E%202025-03-13%2013.13.38%20-%20A%20side-by-side%20comparison%20of%20a%20real%20human%20face%20and%20an%20AI-generated%20human%20face,%20highlighting%20differences%20such%20as%20symmetry%20inconsistencies,%20unnatural%20sk.webp?Expires=1836497644&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=u2ciH6JG~HNPXjvec8ebsY7jLqSOdVUIsw-UJFeXvHvDeXTsOqhmSx7O7O4RqcgBz-wjiC9XwMSXHJv0Htm~3TSi2zcjvc60SXjc0oJd94Nix0-pwko5ApjSmEJ4DhtFhJRytYYGdZzrwmMEJJ6e5EXPH597lHiO99DGYzuW86cLuVTqSVWg5jIBsWmxQmmIiuARxeEp~xwZ9cKWHzWrDePSA2qizXexHKzF9k7b03zv4e6H3tjUUzCvKm1Br4oGDRFPXXuJLdFTSqTO5FvCMtu4R5icVXAz7RQ8bIkmeG8h0Zh1m9ROM1m-7U~Jc5KiFjVN~1HBABYBMuFNgE2VVQ__' width=400>

**Frequency Artifacts & Texture Patterns**

- Real Images follow natural frequency distributions, where textures and fine details (e.g., skin pores, hair strands) have gradual transitions.
- AI-Generated Images may introduce unnatural high-frequency noise or excessive smoothness due to upsampling artifacts.

Reference: Durall et al. (2020) - "Unmasking DeepFakes with Simple Features" (Fourier Transform-based detection) [[arXiv]](https://arxiv.org/abs/1911.00686).

**Inconsistencies in Features**

<img src='https://media-hosting.imagekit.io//f9b2b148e9354048/kids_doing_art_analysis-1024x576.png?Expires=1836498797&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=OlDYuZEBVEJ~sfcGuPsvOiQh~Oyt5~Zy-FnCmagoqMNHaDRk6uDyPerYHno~ftdQoFlgCd7USuiAZX0lwfoG4FIbIc8Aus6VeqkIgGmiBDy8~H4jgE6lBymO4GclfQ27uOL0VS1SiVXySXmv9Z7usj~fKeUxyJEzUcNOsXZzSoeNOwPGSrFBUp0Q8oWPsOodJDbJqODPxPzj1TWZ06m8frRLNOqLROVLmwaRLexrTjJC--HnW-rNpyX0sP8sBsiv70RgKZCqx8BTZySg6hJ~F5BkevoFVYT6XQuorwISOBjEY1KFpLVJn0IkZ9PIWDTeoTnQZGydfQTLEdQZzWbeSg__' width=400>

- Real Faces exhibit natural asymmetry, but AI-generated faces often have perfect symmetry due to generator artifacts.
- GAN-based models, especially StyleGAN, sometimes generate inconsistent eye colors, mismatched earrings, or asymmetric lighting.
- Diffusion models like Stable Diffusion or MidJourney sometimes create distorted hands, extra fingers, or misaligned facial features.

Reference: Nightingale & Farid (2022) - "Detection of GAN-Generated Imagery Using Statistical Inconsistencies" (Facial feature mismatches) [[Paper]](https://library.imaging.org/ei/articles/35/4/MWSF-380).

**Lack of Natural Camera Artifacts**

- Real Photos are captured using physical sensors that introduce sensor noise, lens distortion, and motion blur based on lighting conditions.
- AI-Generated Images are cleaner than real images, often lacking these natural artifacts.

Reference: Wang et al. (2020) - "CNN-Generated Images Are Surprisingly Easy to Spot" (Analyzing sensor noise patterns) [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html).

**Unnatural Shadows & Lighting Effects**

<img src='https://media-hosting.imagekit.io//004acf3849e44008/titantic_analysis-1024x576.png?Expires=1836498853&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=E7XJey-qdiWfBfFp6b83ubPbUuc5bxG-6xplXdLXoV~gyALYpucchwPazTx6KJEN5SBGReD5aMl~qw7iR59D6VbLYDK1Hy8WF7C1LIw6ztunPrOPyBkDmgYAvRjtqPLD8qnNbo28fGEBRiItDO1oJLlFVkj7vOCKhqAGkoA7em93eCtf75ZKtgTHrz3-VYWltAHt-2DWv~CtQ08DeiONihJ9cfIdo3GGpXuiHOV9f26ArCriQ2CxxWT32UDtN5KTu10mPQQaHRVHNOxS9xj7bX73uGfWRXrlHK6-0heOzqHVlc6uNSvFCYN-hRNMqt4tETMy4HCXIrODlICFpvb1nw__' width=400>

- Real Images follow consistent physics-based lighting models, where shadows, highlights, and reflections behave naturally.
- AI-Generated Faces sometimes exhibit misaligned shadows or inconsistent reflections (e.g., teeth and lips reflecting light unrealistically).

Reference: Zhou et al. (2020) - "DeepFake Detection Based on Illumination Inconsistencies" [[IEEE Paper]](https://link.springer.com/chapter/10.1007/978-3-031-06788-4_52).

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

$p(y \mid x, D) \approx \frac{1}{T} \sum_{t=1}^{T} p(y \mid x, W_t)$

where:
-   $p(y \mid x, D)$ is the predicted probability distribution given input $x$ and training data $D$.
-   $T$ is the number of stochastic forward passes (e.g., **200** in our implementation).
-   $W_t$​ represents the model parameters with **dropout applied** during inference.

While Step 6 gives us accuracy, it doesn't provide insights into **how confident the model is** in its predictions. This step adds uncertainty estimation, making our classifier more robust, especially for ambiguous images.
```
def  bayesian_inference(model, feature_vector, num_samples=200):
	model.train() # Keep dropout active for Bayesian inference
	device  =  next(model.parameters()).device
	feature_vector  =  feature_vector.to(device)

	preds  = []
	feature_importances  = []

	# Monte Carlo sampling
	with  torch.no_grad(): # Disable gradient tracking for MC sampling
		for  i  in  range(num_samples):
			# Forward pass
			output  =  model(feature_vector)
			probs  =  torch.nn.functional.softmax(output, dim=1)
			preds.append(probs.cpu())

	# Compute feature importance
	feature_importance  =  compute_feature_importance(model, feature_vector)

	# Aggregate predictions
	preds  =  torch.stack(preds)
	mean_pred  =  preds.mean(dim=0).numpy()
	uncertainty  =  preds.std(dim=0).numpy()

	return  mean_pred, uncertainty, feature_importance
```

- **`model.train()` During Inference**: Unlike normal inference, we keep dropout enabled to introduce randomness in predictions

- **Perform Multiple Stochastic Passes** (`num_samples = 200`): Each pass provides a slightly different output due to the stochastic dropout.
- **Convert Logits to Probabilities**: Use `softmax` to get class probabilities.  
- **Compute Mean Prediction**: The average probability over multiple runs gives a more stable prediction.
- **Estimate Uncertainty**: The standard deviation of predictions provides an uncertainty measure. Higher variance means the model is less confident.

**Why This is Useful?**
-   **Trust & Reliability**: Helps detect cases where the model is uncertain.
-   **Better Decision-Making**: If uncertainty is high, human review may be required.
-   **Adversarial & Edge Case Detection**: Can identify hard-to-classify images.


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

## Visualizing Feature Importance on the Image
This step adds a visual representation to highlight the most influential image regions.

$$
I(x_i) = \left| \frac{\partial f(x)}{\partial x_i} \right|
$$
where:
-   $I(x_i)$ is the importance of the $i$-th feature.
-   $f(x)$ is the model’s prediction function.
-   $\frac{\partial f(x)}{\partial x_i}$​ is the gradient of the output with respect to the feature $x_i​$.
-   Taking the **absolute value** ensures importance is always positive.
```
from matplotlib.patches import Rectangle

def  visualize_feature_importance(image, feature_importance, patch_size=16):
	# Convert image to numpy array if it's a torch tensor
	if  isinstance(image, torch.Tensor):
		image = image.cpu().numpy()

	# Remove batch dimension if present
	if image.ndim == 4: # Shape: (1, C, H, W)
		image = image.squeeze(0) # Remove batch dimension -> (C, H, W)

	# Transpose image to (H, W, C) if necessary
	if image.shape[0] == 3: # Shape: (C, H, W)
		image = np.transpose(image, (1, 2, 0)) # Transpose to (H, W, C)

	# Get image dimensions
	if  len(image.shape) == 2: # Grayscale image (H, W)
		H, W = image.shape
		C = 1  # Single channel
	elif  len(image.shape) == 3: # RGB image (H, W, C)
		H, W, C = image.shape
	else:
		raise  ValueError(f"Unsupported image shape: {image.shape}")

	# Calculate the number of patches along height and width
	H_patches = H // patch_size
	W_patches = W // patch_size

	# Check if the feature importance size matches the patch grid
	if feature_importance.size != H_patches * W_patches:
		print(f"Feature importance size {feature_importance.size} does not match patch grid {H_patches}x{W_patches}.")
		print("Reshaping feature importance to match the patch grid.")
		# Reshape feature_importance to the nearest square dimensions
		size = int(np.sqrt(feature_importance.size))
		feature_importance = feature_importance[:size * size].reshape(size, size)
		H_patches, W_patches = size, size # Update patch grid dimensions

	# Create a figure to display the image
	plt.figure(figsize=(10, 5))
	if C == 1: # Grayscale image
	plt.imshow(image, cmap='gray')
	else: # RGB image
	plt.imshow(image)

	# Overlay patches with feature importance
	for i in  range(H_patches):
		for j in  range(W_patches):
			importance = feature_importance[i, j]
			# Clip alpha to the range [0, 1]
			alpha = np.clip(importance, 0, 1)
			# Draw a rectangle for each patch
			rect = Rectangle(
			(j * patch_size, i * patch_size), # (x, y) of the patch
			patch_size, patch_size, # Width and height of the patch
			linewidth=1, edgecolor='r', facecolor='yellow', alpha=alpha
			)
			plt.gca().add_patch(rect)

	plt.colorbar(label='Feature Importance')
	plt.title('Feature Importance Visualization (Patches)')
	plt.axis('off')
	plt.show()
```
- **Preprocesses Image for Visualization**: Converts PyTorch tensors into NumPy arrays and ensures correct formatting (`(H, W, C)`).  
- **Divides Image into Patches**: Uses `patch_size = 16` to match the **DINO ViT’s feature extraction resolution**.  
- **Maps Feature Importance onto Image**:
 -- Uses **alpha blending** to overlay feature importance patches.
 -- More important patches are highlighted more strongly.  
 -- **Handles Reshaping Issues** – Adjusts feature importance dimensions if they do not match the patch grid.  
 
## Classifying real-world images
```
def  preprocess_image(image_path):
	image = Image.open(image_path).convert("RGB")
	image = transform(image).unsqueeze(0) # Add batch dimension
	return image

def  predict_image(image_path, uncertainty_threshold_multiplier=1.0, top_n=10, patch_size=16):
	# Preprocess Image & Extract Features
	image = preprocess_image(image_path)
	features = extract_features(image)

	# Predict Class (Standard Inference)
	classifier.eval()
	with torch.no_grad():
		output = classifier(features)
		probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
	class_idx = np.argmax(probabilities) # Get predicted class (AI or Real)
	confidence = probabilities[class_idx] # Get confidence score
	  
	print(f"Predicted Class: {dataset.classes[class_idx]}")
	print(f"Confidence: {confidence:.4f}")

	# Run Bayesian Inference (MC Dropout)
	mean_pred, uncertainty, feature_importance = bayesian_inference(classifier, features)

	# Compute Uncertainty for Predicted Class
	uncertainty = uncertainty.squeeze() # Ensure uncertainty is 1D

	# Handle unexpected uncertainty shape
	if uncertainty.ndim > 1  or  len(uncertainty) <= class_idx:
		print("Unexpected Uncertainty Shape, Defaulting to Mean Uncertainty")
	uncertainty_score = float(np.mean(uncertainty)) # Default to mean if incorrect shape
	else:
		uncertainty_score = float(uncertainty[class_idx]) # Extract as a scalar

	print(f"Bayesian Uncertainty: {uncertainty_score:.4f}")

	# Visualize Feature Importance
	visualize_feature_importance(image, feature_importance, patch_size=patch_size) # Visualize important patches

	return dataset.classes[class_idx], confidence, uncertainty_score, feature_importance
```
Finally, we build a full prediction pipeline for classifying new images. This function allows us to input an image, classify it, measure uncertainty, and visualize what influenced the model’s decision.

#### `preprocess_image(image_path)`
- Loads an image from disk using `PIL.Image`.  
- Converts it to RGB format.  
- Applies the same `transform` pipeline used in training.  
- Adds a **batch dimension** (`unsqueeze(0)`) so it matches model expectations.

#### `predict_image(image_path, uncertainty_threshold_multiplier=1.0, top_n=10, patch_size=16)`
- Preprocess Image & Extract Features using `preprocess_image(image_path)` and `extract_features(image)` respectively

- Standard Inference (Baseline Prediction): Performs **softmax activation** to obtain class probabilities and extracts the predicted class index and confidence score.

- Bayesian Inference for Uncertainty Estimation: Calls `bayesian_inference(classifier, features)` to get Mean prediction over multiple MC Dropout runs (`mean_pred`), Standard deviation across runs (`uncertainty`), and Gradient-based importance scores (`feature_importance`).

- Compute Uncertainty Score: Extract uncertainty for the predicted class

## Occlusion Sensitivity
```
def  occlusion_sensitivity(image, model, patch_size=16, stride=8):
	image_np = image.squeeze().permute(1, 2, 0).cpu().numpy() # Convert to numpy
	h, w, _ = image_np.shape

	# Initialize the sensitivity map
	sensitivity_map = np.zeros((h, w))

	# Loop through the image with sliding window
	for i in  range(0, h - patch_size, stride):
		for j in  range(0, w - patch_size, stride):
			# Create an occluded copy of the image
			occluded_image = image.clone()
			occluded_image[:, :, i:i+patch_size, j:j+patch_size] = 0  # Occlude the patch

			# Extract features from occluded image
			occluded_features = extract_features(occluded_image)

			# Perform inference
			with torch.no_grad():
				output = model(occluded_features)
				prob = torch.nn.functional.softmax(output, dim=1).cpu().numpy()

			# Calculate the sensitivity score for the current patch
			sensitivity_map[i:i+patch_size, j:j+patch_size] = prob[0, 1] # Sensitivity score for class 1 (AI-Generated)

	return sensitivity_map
```

**Sliding Window Occlusion**: Moves a patch across the image, setting pixels to zero (blackout occlusion).
**Feature Extraction & Prediction**: The occluded image is passed through DINOv2 and the classifier to check how the prediction changes.  
**Sensitivity Mapping**: If occlusion lowers AI-generated confidence, that region is important for classification.

## Final Output
```
image_path = 'PATH_TO_IMAGE'
predict_image(image_path)
```

<img src='https://media-hosting.imagekit.io//d877872a9bfb442d/Screenshot%202025-03-12%20at%2011.52.31%E2%80%AFPM.png?Expires=1836449582&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=pO~duEL3ILzAQJiAVR~5WOha4jEwCsDMcCDEGXuaTCWH2DLLikLP5FeUWv7dOO375wj~uHbefDlPqmDK1EgHP-0ldMAhqZTeU1P3PgBWQQDTG63y1B3FF-6DI66iIq0LIsn3gzkW0636Tx6rFq2WaWXrku2ohOBfo35n~w8HjAJYaif-qj-3hc1oCTKXOy23yHvPxSsCNXCvQd7Xgo2eEQpn6WcqV29~z~EiPk5O1eIhSFmB1PlCGGlTgzGEdeG3RX3zjYjW1ku96QLQB6iUhwQWb5ECjVjQSIhEHOdNfiJon7zI65yBVhUqEec~mwEfDfFMWlPndTiJs4KpnL0mXQ__' width=400>

<img src='https://media-hosting.imagekit.io//4004cfe517c64a79/Screenshot%202025-03-13%20at%2012.16.47%E2%80%AFPM.png?Expires=1836494220&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=0sQ~sVNu~zxMzOfrKy1uhivckKSJN6acXDKfimrNS9C~rWvdQTbCxNoSVKBO3lrQ2chDHbOYQy6BuTXGqt09twx3r2cYZ5qryagQ4aNrhh5EBeVSBmN6XekKKVJ54MB2iDx3XWc4qy6x7jWdoJWcurtL1lIkkO986LkNEZBhAySLr9X5zoxaUiJLqnZqb6T72VGKvOho-HPLeq1qdyhpNlzF2kbqPoWY7aopjswC2jHA0hxOqBocVf8t2XmSrdPmW7bp1yKUSvL~Q-TxSkZ1-1-uVSTKzQFhAESJfbLyYa-3uVwybk7LcbBGVP-B46kntm-zfqP7bbGW8rO1ehm5hw__' width=400>

## Future Scope
#### Improve Model Robustness Against Advanced AI-Generated Images
Train on a Larger and More Diverse Dataset: Current models like Stable Diffusion 3 and MidJourney v6 generate highly realistic faces. Expanding the dataset to include newer synthetic images will enhance detection capabilities.

#### Multi-Modal Detection (Beyond Images)
- Extend Model to Video Deepfake Detection: Apply the classification framework to **frame-by-frame video analysis** to detect AI-generated videos.
- Combine Image + Text Metadata: AI-generated content often comes with metadata traces. Integrating text-based cues can enhance classification.

#### Explainability & Model Interpretability Enhancements
Improve Feature Importance Visualization: Instead of using only gradient-based importance, explore **Grad-CAM** or **Saliency Graphs**
