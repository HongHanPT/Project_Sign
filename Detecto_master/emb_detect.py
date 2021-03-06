import torch
from detecto import core, utils, visualize
import cv2
import os
from ClassifySign.classifysign import classify

image = utils.read_image('A.jpg')
model = core.Model.load('model_weights.pth', ['HexagonSign', 'RhombusSign'])
#model.summary() # Dang su dung ham tu viet, co the dung torchsummary: pip install torchsummary

# dim = (480, 320)
# image = cv2.imread(filename)
# image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
try:
    torch.cuda.set_device(0)
except:
    print("Khong dung duoc gpu")
def detectCamera(iou_threshold):
        try:
           vc = cv2.VideoCapture(-1,cv2.CAP_V4L2)
           print("Loi lan 1")
        except:
            vc = cv2.VideoCapture(0, cv2.CAP_V4L2)
            print("Loi lan 2")
        try:
            vc = cv2.VideoCapture(0)
            print("Loi lan 3")
        except:
            print("Loi tiep")

        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        import time
        count =0
        stt_show =0
        while rval:
            dim = (480, 640)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
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
                compare = scores > iou_threshold
                j = 0
                boxes_new = []
                labels_new = []
                scores_new = []
                for i in range(int(len(compare))):
                  if compare[i] == True:
                          boxes_new.append(boxes[i])
                          labels_new.append(labels[i])
                          scores_new.append(scores[i])
                          j = j + 1
                if j > 0:
                  stt_show=1
                  # convert list to torch tensor
                  boxes_new = torch.stack(boxes_new)
                  # Plot each box
                  for i in range(boxes_new.shape[0]):
                          box = boxes_new[i]

                          score_class, classes = classify(frame, box)
                          if labels_new[i] == "HexagonSign":
                              rH = scores_new[i]
                              rR =0
                          elif labels_new[i] == "RhombusSign":
                              rH =0
                              rR = scores_new[i]
                          if (classes == 'Slow'):
                              check = score_class * rR
                          if (classes == 'Fast'):
                              check = score_class * rR
                          if (classes == 'Start'):
                              check = score_class * rH
                          if (classes == 'Stop'):
                              check = score_class * rH
                          cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                          if(check > 0.6):
                            cv2.putText(frame, '{}'.format(classes), (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0),2,cv2.LINE_AA)
                          elif check < 0.6:
                            cv2.putText(frame, 'Khong xac dinh', (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0),2,cv2.LINE_AA)
                cv2.imshow("preview", frame)


            rval, frame = vc.read()
            count = count +1

            if key == 27:  # exit on ESC
              break
        vc.release()
        cv2.destroyWindow("preview")
def detectImages(img_path, iou_threshold):
        for filename in os.listdir(img_path):
            dim = (320, 480)
            image = cv2.imread(os.path.join(img_path,filename))
            #print(image.shape)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("preview", image)
            labels, boxes, scores = model.predict(image)
            compare = scores > iou_threshold
            j = 0
            #print(type(boxes))
            boxes_new = []
            labels_new = []
            scores_new = []
            for i in range(int(len(compare))):
                if compare[i] == True:
                    boxes_new.append(boxes[i])
                    labels_new.append(labels[i])
                    scores_new.append(scores[i])
                    j = j + 1

            if j > 0:
                stt_show = 1
                # convert list to torch tensor
                boxes_new = torch.stack(boxes_new)
                # Plot each box
                for i in range(boxes_new.shape[0]):
                    box = boxes_new[i]

                    score_class, classes = classify(image, box)
                    if labels_new[i] == "HexagonSign":
                        rH = scores_new[i]
                        rR = 0
                    elif labels_new[i] == "RhombusSign":
                        rH = 0
                        rR = scores_new[i]
                    if (classes == 'Slow'):
                        check = score_class * rR
                    if (classes == 'Fast'):
                        check = score_class * rR
                    if (classes == 'Start'):
                        check = score_class * rH
                    if (classes == 'Stop'):
                        check = score_class * rH
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                    if (check > 0.6):
                        cv2.putText(image, '{} {}'.format(classes, check), (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)
                    elif check < 0.6:
                        cv2.putText(image, 'Khong xac dinh', (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("preview", image)
            cv2.imwrite(os.path.join(img_path, "_"+filename), image)
if __name__ == '__main__':
    detectImages(r'C:\Users\handi\OneDrive\Documents\Project_Sign\Detecto_master\images', 0.5)
    #detectCamera(0.1)