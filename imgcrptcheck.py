'''
@author : fharookshaik
'''

import os
from os.path import sep
from PIL import Image
from tqdm import tqdm
import cv2 as cv

IMG_PATH = r"F:\Training Data\public_image_set"

def isCorrupted(img):
    try:
        im = Image.open(img)
        im.verify()
        return (False, None)

    except Exception as e:
        return (True,e)

f = open('Corrupted-Images-ALL.txt','a')

def returnCorruptedImages():
    count = 0
    for img in tqdm(os.listdir(IMG_PATH)):
        impath = os.path.join(IMG_PATH,img)
        isCorrupt,e = isCorrupted(impath)
        if isCorrupt:
            count += 1
            # print('\n',img,e,'\n')
            f.write('\n {}: {}'.format(img,e))
    return count

f.write('count = ' +str(returnCorruptedImages()))        
f.close()