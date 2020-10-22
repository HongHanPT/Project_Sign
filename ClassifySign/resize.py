from PIL import Image
import os
 
from os.path import basename
 
def resizeImages(baseDir):
    basewidth = 32
    for filename in os.listdir(baseDir):
        filenameOnly, file_extension = os.path.splitext(filename)
        # print (file_extension)
        if (file_extension in [".jpg", '.png']):
            filepath = baseDir + os.sep + filename
            img = Image.open(filepath)
            # wpercent = (basewidth/float(img.size[0]))
            # hsize = int((float(img.size[1])*float(wpercent)))
            hsize = 32
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.save(filepath)
            print (filenameOnly, "Done")
    print('Done')
         
# Usage
baseDir = "D:\TPA\Projects\Vision\ClassifySign\python_process_image\All_Croped"
resizeImages(baseDir)