import requests
from bs4 import BeautifulSoup
import os
from PIL import Image
import cv2
import numpy as np

def make_transparent_white(img):
    img = img.convert("RGBA")
    # Load the image and make into Numpy array
    img = np.array(img)
    # Make image transparent white anywhere it is transparent
    img[img[...,-1]==0] = [255,255,255,0]

    # Make back into PIL Image and convert to RGB
    return Image.fromarray(img)


def make_white_transparent(img):
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    return img

def make_RGB(img):
    img = img.convert("RGB")
    return img

def make_RGBA(img):
    img = img.convert("RGBA")
    return img

def make_black(img):
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((0,0,0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img

if __name__ == "__main__":
    dirs = os.listdir(os.getcwd())

    for img_path in dirs:
        try:
            if '.png' not in img_path and '.jpeg' not in img_path:
                os.remove(os.path.join(os.getcwd(),img_path))
            elif '.png' not in img_path:
                img = Image.open(img_path)
                img.save(f"{img_path[:-4]}png")
                os.remove(os.path.join(os.getcwd(),img_path))
                print(img_path)
        except:
            pass
