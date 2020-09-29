import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import shutil  

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'

path = '/mnt/projects/bilvlda/dataset/tiny-imagenet-200/val'

with open(os.path.join(path, VAL_ANNOTATION_FILE), 'r') as fp:
    for line in fp.readlines():
        terms = line.split('\t')
        file_name, directory = terms[0], terms[1]
        if not os.path.exists(os.path.join(path, directory, file_name)):
            dest = shutil.move(os.path.join(path, file_name), os.path.join(path, directory))