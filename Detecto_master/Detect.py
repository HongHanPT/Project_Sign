import torch
from detecto import core, utils, visualize

image = utils.read_image('A.jpg')
model = core.Model.load('model_weights.pth', ['HexagonSign', 'RhombusSign'])

labels, boxes, scores = model.predict(image)
#visualize.show_labeled_image(image, boxes, labels)
print(scores)
#print(boxes)
a = scores > 0.7
print(a)
j=0
print(type(boxes))
boxes_new = []
labels_new=[]
for i in range(int(len(a))):
  if a[i]== True:

    boxes_new.append(boxes[i])
    labels_new.append(labels[i])
    j = j +1
    print(labels[i])
    print(boxes[i])
    print(scores[i])
print(type(boxes[0]))

#convert list to torch tensor
boxes_new = torch.stack(boxes_new)
print(boxes_new.shape)
print((labels_new))
visualize.show_labeled_image(image, boxes_new, labels_new)