import numpy as np

from PIL import Image
from datasets.camvid.labels import get_classes

def generate_image(output, image_channels=3, label_path='./datasets/camvid/labels_v2.txt'):
    labels, colors, color2id = get_classes(label_path)

    row, column = output.shape
    img_arr = np.zeros((row,column,image_channels))
    for r in range(row):
        for c in range(column):
            id = output[r,c] 
            rr, gg, bb = colors[id]
            img_arr[r,c,:] = [rr,gg,bb]
    return img_arr.astype(np.uint8)
