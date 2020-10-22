import torch
from detecto import core, utils, visualize
import cv2

image = utils.read_image('A.jpg')
model = core.Model.load('model_weights.pth', ['HexagonSign', 'RhombusSign'])


vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
import time
while rval:
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    start = time.time()
    labels, boxes, scores = model.predict(frame)
    stop = time.time()
    print(stop - start)
    a = scores > 0.7
    j = 0
    boxes_new = []
    labels_new = []
    for i in range(int(len(a))):
      if a[i] == True:
        boxes_new.append(boxes[i])
        labels_new.append(labels[i])
        j = j + 1
    if j > 0:
      # convert list to torch tensor
      boxes_new = torch.stack(boxes_new)
      print("ad")
      # Plot each box
      for i in range(boxes_new.shape[0]):
          box = boxes_new[i]
          cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
          if labels:
            cv2.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')


    rval, frame = vc.read()

    if key == 27:  # exit on ESC
      break
cv2.destroyWindow("preview")

