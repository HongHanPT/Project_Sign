import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import transforms
from detecto import core, utils, visualize

# Specify a list of transformations for our dataset to apply on our images
transform_img = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(800),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

dataset = core.Dataset('train.csv', 'images/', transform=transform_img)

# dataset[i] returns a tuple containing our transformed image and
# and a dictionary containing label and box data
image, target = dataset[0]

# Show our image along with the box. Note: it may
# be colored oddly due to being normalized by the
# dataset and then reverse-normalized for plotting
visualize.show_labeled_image(image, target['boxes'], target['labels'])

# Create our validation dataset
val_dataset = core.Dataset('val.csv', 'images/')

# Create the loader for our training dataset
# You can skip this step

# loader = core.DataLoader(dataset, batch_size=2, shuffle=True) --Having error
loader = core.DataLoader(val_dataset, batch_size=2, shuffle=True)

# Create our model, passing in all unique classes we're predicting
# Note: make sure these match exactly with the labels in the XML/CSV files!
model = core.Model(['HexagonSign', 'RhombusSign'])

# Train the model!
losses = model.fit(loader, val_dataset, epochs=10, verbose=True) # If you skip the loader step, replace loader with dataset (Here is val_dataset because of having a error in dataset)
model.save('model_weights.pth')
# Plot the accuracy over time
plt.plot(losses)
plt.show()