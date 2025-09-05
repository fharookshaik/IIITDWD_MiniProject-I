'''
@author: fharookshaik
'''

import sys
import pandas as pd  
import os
import shutil as sh
from tqdm import tqdm
from sklearn.utils import shuffle
import argparse
from pathlib import Path

# Existing Dataset paths
TRAIN_CSV_PATH = Path('DATASET\TRAIN\multimodal_train.csv')
TEST_CSV_PATH = Path('DATASET\TEST\multimodal_test.csv')
VALIDATE_CSV_PATH = Path('DATASET\VALIDATE\multimodal_validate.csv')

IMG_FOLDER_PATH = Path("F:\Training Data\public_image_set")



def _set_new_dataset_paths(percent,dest_dir_path=''):
    
    global DEST_DIR_PATH
    global NEW_TRAIN_DIR_PATH
    global NEW_TRAIN_CSV_PATH
    global NEW_TRAIN_IMG_PATH
    global NEW_TEST_DIR_PATH
    global NEW_TEST_CSV_PATH
    global NEW_TEST_IMG_PATH
    global NEW_VALIDATE_DIR_PATH
    global NEW_VALIDATE_CSV_PATH
    global NEW_VALIDATE_IMG_PATH
    
    if dest_dir_path=='':
        DEST_DIR_PATH = os.getcwd() 
    else:
        DEST_DIR_PATH = dest_dir_path
    
    #NEW Dataset paths 
    NEW_TRAIN_DIR_PATH = os.path.join(DEST_DIR_PATH,'NEW_DATASET_{}/TRAIN'.format(percent))
    NEW_TEST_DIR_PATH = os.path.join(DEST_DIR_PATH,'NEW_DATASET_{}/TEST'.format(percent))
    NEW_VALIDATE_DIR_PATH = os.path.join(DEST_DIR_PATH,'NEW_DATASET_{}/VALIDATE'.format(percent))

    NEW_TRAIN_CSV_PATH = os.path.join(NEW_TRAIN_DIR_PATH,'multimodal_train.csv')
    NEW_TEST_CSV_PATH = os.path.join(NEW_TEST_DIR_PATH,'multimodal_test.csv')
    NEW_VALIDATE_CSV_PATH = os.path.join(NEW_VALIDATE_DIR_PATH,'multimodal_validate.csv')

    NEW_TRAIN_IMG_PATH = os.path.join(NEW_TRAIN_DIR_PATH,'IMAGES')
    NEW_TEST_IMG_PATH = os.path.join(NEW_TEST_DIR_PATH,'IMAGES')
    NEW_VALIDATE_IMG_PATH = os.path.join(NEW_VALIDATE_DIR_PATH,'IMAGES')


def _create_path(path):
    try:
        if os.path.exists(path):
            return True
        else:
            print('PATH {} NOT FOUND. CREATING A NEW ONE'.format(path))
            os.makedirs(path)
    except Exception as e:
        print('Exception creating path : ' + str(e))

def _check_path_exists(path,create_new=False):
    try:
        if os.path.exists(path):
            print('FOUND {}'.format(path))
            return True
        else:
            if create_new:
                _create_path(path)
                _check_path_exists(path)

    except Exception as e:
        print('Exception Checking path exists: ' + str(e))

# def get_new_dataset_by_numOfRows(df,numOfRows,csv_path):
#     df = shuffle(df)    
#     new_df = df.iloc[:numOfRows,:]
#     new_df.to_csv(csv_path)
#     return new_df

def _get_df_by_noOfRows(df,numOfRows):
    df = shuffle(df)
    return df.iloc[:numOfRows,:]

def _get_new_dataset_by_percent(df,percent,csv_path):
    grouped_df = df.groupby(['2_way_label'])
    true_df = grouped_df.get_group(1)
    false_df = grouped_df.get_group(0)
    numOfRows = int(df.shape[0] * (percent / (100 * 2)))
    # new_df = df.iloc[:numOfRows,:]
    new_df = pd.concat([_get_df_by_noOfRows(true_df,numOfRows),
                        _get_df_by_noOfRows(false_df,numOfRows)])
    new_df.to_csv(csv_path)
    return new_df

def _get_images_to_folder(df,img_folder_path='',newfolderName=''):
    # Gets the values of 'id' which resembles the image in the folder 
    img_col = df.iloc[:,[df.columns.get_loc('id')]].values
    print(img_col.shape)
    try:
        if os.path.exists(newfolderName):
            pbar = tqdm(img_col)
            for i in pbar:
                img = os.path.join(img_folder_path, '{}.jpg'.format(i[0]))
                dest_img = os.path.join(newfolderName, '{}.jpg'.format(i[0]))
                if os.path.exists(img):
                    pbar.set_description('Copying {}.jpg'.format(i[0])) 
                    sh.copyfile(img,dest_img)
                    # print('Copying ', i[0])
    except Exception as e:
        print('Exception: ' +  str(e))

if __name__ == '__main__':
    
    try:
        parser = argparse.ArgumentParser(description='Program to extract Custom size dataser from the original dataset by percent.')
        parser.add_argument('-p','--percent',type=int,default=10,help='Enter the %% of dataset needed.Default = 10')

        args = parser.parse_args()
        
        if args.percent:
            percent = args.percent
        # elif args.destination_path:
        #     DEST_PATH = args.destination_path
        DEST_PATH = os.getcwd()
        _set_new_dataset_paths(percent,DEST_PATH)
        # Check if original dataset path exists.
        _check_path_exists(TRAIN_CSV_PATH)
        _check_path_exists(TEST_CSV_PATH)
        _check_path_exists(VALIDATE_CSV_PATH)
        _check_path_exists(IMG_FOLDER_PATH)


        # Check if new dataset path exists. Otherwise create new respective paths.
        _check_path_exists(NEW_TRAIN_DIR_PATH,create_new=True)
        _check_path_exists(NEW_TEST_DIR_PATH,create_new=True)
        _check_path_exists(NEW_VALIDATE_DIR_PATH,create_new=True)
        _check_path_exists(NEW_TRAIN_IMG_PATH,create_new=True)
        _check_path_exists(NEW_TEST_IMG_PATH,create_new=True)
        _check_path_exists(NEW_VALIDATE_IMG_PATH,create_new=True)

        # Loading Exiisting dataset csv paths
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        test_df = pd.read_csv(TEST_CSV_PATH)
        validate_df = pd.read_csv(VALIDATE_CSV_PATH)

        # Printing the dataset dimensions
        print('-'*50,' '*10 + 'DATASET DIMENSIONS' + ' '*10,'-'*50,sep='\n')
        print('TRAIN_DATASET : ' ,train_df.shape)
        print('TEST_DATASET : ' ,test_df.shape)
        print('VALIDATE_DATASET : ' ,validate_df.shape)

        # GET DATA BY PERCENT -> Uncomment the following lines for getting the data by percent

        print('-'*50)
        print(' '*10 + 'GETTING {}% DATA'.format(percent) + ' '*10)
        print('-'*50)
        new_train_df = _get_new_dataset_by_percent(train_df,percent,NEW_TRAIN_CSV_PATH)
        new_test_df = _get_new_dataset_by_percent(test_df,percent,NEW_TEST_CSV_PATH)
        new_validate_df = _get_new_dataset_by_percent(validate_df,percent,NEW_VALIDATE_CSV_PATH)

        # GET DATA BY No.Of.Rows
        
        # numOfRows = 10000
        # new_train_df = get_new_dataset_by_numOfRows(train_df,numOfRows,NEW_TRAIN_CSV_PATH)
        # new_test_df = get_new_dataset_by_numOfRows(test_df,numOfRows,NEW_TEST_CSV_PATH)
        # new_validate_df = get_new_dataset_by_numOfRows(validate_df,numOfRows,NEW_VALIDATE_CSV_PATH)

        _get_images_to_folder(new_train_df,IMG_FOLDER_PATH,NEW_TRAIN_IMG_PATH)
        _get_images_to_folder(new_test_df,IMG_FOLDER_PATH,NEW_TEST_IMG_PATH)
        _get_images_to_folder(new_validate_df,IMG_FOLDER_PATH,NEW_VALIDATE_IMG_PATH)
    
    except Exception as e:
        print('Error : ',str(e))
        sys.exit(0)
