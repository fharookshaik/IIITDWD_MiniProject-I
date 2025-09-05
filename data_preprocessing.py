'''
@author: fharookshaik

Status : Only Image Pre-processing modules were implemented. Text pre-processing needs to be implemented yet.

'''

# Imports
import sys
import os
import pandas as pd
import numpy as np
import cv2 as cv
from pathlib import Path
import tqdm
import PIL
from PIL import Image
from sklearn.utils import shuffle
from pathlib import Path
from alive_progress import alive_bar
import shutil as sh


# Default Image size and Max Image pixels
Image.MAX_IMAGE_PIXELS = 933120000
IMG_SIZE = (120,120)

# DATASET DIRECTORY PATHS
DATASET_DIR = Path('NEW_DATASET_20')

# TRAIN DATA PATH
TRAIN_CSV_PATH = os.path.join(DATASET_DIR,'TRAIN','multimodal_train.csv')
TRAIN_IMG_PATH = os.path.join(DATASET_DIR,'TRAIN','IMAGES')

# TEST DATA PATH
TEST_CSV_PATH = os.path.join(DATASET_DIR,'TEST','multimodal_test.csv')
TEST_IMG_PATH = os.path.join(DATASET_DIR,'TEST','IMAGES')

# VALIDATE DATA PATH
VALIDATE_CSV_PATH = os.path.join(DATASET_DIR,'VALIDATE','multimodal_validate.csv')
VALIDATE_IMG_PATH = os.path.join(DATASET_DIR,'VALIDATE','IMAGES')

# OUTPUT PATH
OUTPUT_DIR = os.path.join(DATASET_DIR,'PROCESSED_DATA')
OUTPUT_TRAIN_PATH = os.path.join(OUTPUT_DIR,'train_data')
OUTPUT_TEST_PATH = os.path.join(OUTPUT_DIR,'test_data')
OUTPUT_VALIDATE_PATH = os.path.join(OUTPUT_DIR,'validate_data')


def _isImgCorrupted(img):
    try:
        im = Image.open(img)
        im.verify()
        return False
 
    except FileNotFoundError:
        return True
    
    except SyntaxError:
        return True
    
    except PIL.UnidentifiedImageError:
        return True

def _preprocessImageData(main_df,IMG_PATH):

    df = main_df[['id','2_way_label']]

    X = []
    y = []
    id = []
 
    # pbar = alive_bar(df.shape[0],title='Processing Image Data')

    with alive_bar(df.shape[0],title='Processing Image Data') as pbar:
        for i, row in df.iterrows():
            img, label = row.tolist()
            impath = os.path.join(IMG_PATH, f'{img}.jpg')
            try:
                if not _isImgCorrupted(impath):
                    imarray = cv.imread(impath, cv.IMREAD_GRAYSCALE)
                    if imarray is not None:
                        new_imarray = cv.resize(imarray, IMG_SIZE)
                        X.append(new_imarray)
                        y.append(label)
                        id.append(img)
    
            except Exception as e:
                print(f"{img}: {e}")
            pbar()
    
    # Converting X, y,id to numpy array
    X = np.array(X).reshape(-1,120,120,1)
    y = np.array(y)
    id = np.array(id)
 
    # Normalize X
    X = X / 255.0
 
    return X,y,id

def _store_data(file_path,file):
    np.savez_compressed(file_path,x=file[0],y=file[1],id=file[2])


if __name__ == '__main__':

    if os.path.exists(OUTPUT_DIR):
        sh.rmtree(OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)
    else:
        os.mkdir(OUTPUT_DIR)
    
    train_df = pd.read_csv(TRAIN_CSV_PATH,low_memory=False)
    test_df = pd.read_csv(TEST_CSV_PATH,low_memory=False)
    validate_df = pd.read_csv(VALIDATE_CSV_PATH,low_memory=False)

    # Processing Train Data
    print('-'*20,'Processing Train Data','-'*20)
    Train_Data = _preprocessImageData(train_df,TRAIN_IMG_PATH)
    _store_data(OUTPUT_TRAIN_PATH,Train_Data)

    # Processing Test Data
    print('-'*20,'Processing Test Data','-'*20)
    Test_Data = _preprocessImageData(test_df,TEST_IMG_PATH)
    _store_data(OUTPUT_TEST_PATH,Test_Data)

    # Processing Validate Data
    print('-'*20,'Processing Validate Data','-'*20)
    Validate_Data = _preprocessImageData(validate_df,VALIDATE_IMG_PATH)
    _store_data(OUTPUT_VALIDATE_PATH,Validate_Data)