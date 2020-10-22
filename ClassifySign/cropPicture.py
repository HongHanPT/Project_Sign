from PIL import Image
import os
from xml.dom.minidom import parse
from xml.dom import minidom
import xml.etree.ElementTree as ET
def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    #cropped_image.show()
 
 
if __name__ == '__main__':
    print('A')
    directory = 'D:\TPA\Projects\Vision\ClassifySign\python_process_image\All'
    path = 'D:\TPA\Projects\Vision\ClassifySign\python_process_image'
    for filename in os.listdir(directory):
        onlyName, type = filename.split('.')
        # print(path+'\All\\'+onlyName+'.xml')
        if type == 'jpg':
                print(onlyName)
                xmldoc = minidom.parse(path+'\All\\'+onlyName+'.xml')
                Tree = ET.parse(path+'\All\\'+onlyName+'.xml')
                root = Tree.getroot()
                coords=(int(root.find('object')[4][0].text),int(root.find('object')[4][1].text),int(root.find('object')[4][2].text),int(root.find('object')[4][3].text))
                print(coords)

                crop(path+'\All\\'+onlyName + '.jpg', coords, path+'\All_Croped\\'+onlyName+ '_Croped.jpg')