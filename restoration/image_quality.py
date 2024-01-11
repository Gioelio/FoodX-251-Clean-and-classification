import glob
import os
import cv2 as cv
from tqdm import tqdm_notebook as tqdm
from preprocessing import pipeline
from brisque import brisque


def build_restored_dataset(source_path, destination_path):
    os.makedirs(destination_path, exist_ok=True)
    q = brisque.BRISQUE(url=False)
    for file in tqdm(glob.glob(source_path + '*.jpg')):
        image = cv.imread(file)
        image = pipeline(image)
        quality = q.score(image)
        if quality < 85:
            cv.imwrite(destination_path + os.path.basename(file), image)
