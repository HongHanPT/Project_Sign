import matplotlib.pyplot as plt

from detecto import core

# Create our train dataset
dataset = core.Dataset('train.csv', 'images/')


# Create our validation dataset
val_dataset = core.Dataset('val.csv', 'images/')

# Create the loader for our training dataset
# You can skip this step

# loader = core.DataLoader(dataset, batch_size=2, shuffle=True)
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