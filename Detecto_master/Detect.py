import torch
from detecto import core, utils
import cv2
import sys
import os
from absl import flags
from absl import app
from detecto import core, utils, visualize
import cv2

image = utils.read_image('A.jpg')
model = core.Model.load('model_weights.pth', ['HexagonSign', 'RhombusSign'])

flags.DEFINE_string('type','images', 'Type Detection')
flags.DEFINE_float('iou_thr', '0.6', 'Threshold of Detection')
flags.DEFINE_string('img_dir', '', 'Path to diretory of images')
FLAGS = flags.FLAGS
# dim = (480, 320)
# image = cv2.imread(filename)
# image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def main(argv):
    if FLAGS.type == 'webcam':
        vc = cv2.VideoCapture(-1,cv2.CAP_V4L2)

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        import time
        count =0
        stt_show =0
        while rval:
            if(stt_show ==0):
                cv2.imshow("preview", frame)
            key = cv2.waitKey(20)
            if count>=60:
                count=0
                start = time.time()
                labels, boxes, scores = model.predict(frame)
                stop = time.time()
                stt_show=0
                print("fps: {}".format(1/(stop - start)))
                compare = scores > FLAGS.iou_threshold
                j = 0
                boxes_new = []
                labels_new = []
                for i in range(int(len(compare))):
                  if compare[i] == True:
                          boxes_new.append(boxes[i])
                          labels_new.append(labels[i])
                          j = j + 1
                if j > 0:
                  stt_show=1
                  # convert list to torch tensor
                  boxes_new = torch.stack(boxes_new)
                  print("ad")
                  # Plot each box
                  for i in range(boxes_new.shape[0]):
                          box = boxes_new[i]
                          cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                          print(labels_new[i])
                          print(scores)
                          cv2.putText(frame, '{}'.format(labels_new[i]), (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0),2,cv2.LINE_AA)
                cv2.imshow("preview", frame)


            rval, frame = vc.read()
            count = count +1

            if key == 27:  # exit on ESC
              break
        vc.release()
        cv2.destroyWindow("preview")
    elif FLAGS.type == 'images':
        print(isinstance(FLAGS.img_path, str))
        assert FLAGS.img_path, '`img_path` is missing.'
        for filename in os.listdir(FLAGS.img_path):
            dim = (480, 320)
            image = cv2.imread(filename)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            labels, boxes, scores = model.predict(image)
            compare = scores > FLAGS.iou_threshold
            j = 0
            boxes_new = []
            labels_new = []
            for i in range(int(len(compare))):
                if compare[i] == True:
                    boxes_new.append(boxes[i])
                    labels_new.append(labels[i])
                    j = j + 1
            if j > 0:
                # convert list to torch tensor
                boxes_new = torch.stack(boxes_new)
                # Plot each box
                for i in range(boxes_new.shape[0]):
                    box = boxes_new[i]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                    print(labels_new[i])
                    print(scores)
                    cv2.putText(image, '{}'.format(labels_new[i]), (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("preview", image)


if __name__ == '__main__':
    app.run(main)
