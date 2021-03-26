import os
import shutil
from PIL import Image
import cv2
import numpy as np
import ntpath

def unpacking(path,new_path):
    for pokemon in os.listdir(path):
        # if it there are multiple versions of a pokemon saved in a subdirectories
        # get every version of that pokemon and copy it to the directory with
        # all data (for the alternative_folder and sprites_folder)
        if os.path.isdir(os.path.join(path,pokemon)):
            for image in os.listdir(os.path.join(path,pokemon)):
                print(f"Copying {image}")
                # join the current folder with the pokemon folder and the respecitve image
                pokemon_path = os.path.join(path, pokemon, image)
                # copy it to the new folder
                shutil.copy(pokemon_path, new_path)
        else:
            # if there is only one image per pokemon copy that image to that newer folder
            # (for 2D and 3D folders)
            for image in os.listdir(os.path.join(path,image)):
                print(f"Copying {image}")
                pokemon_path = os.path.join(path, image)
                shutil.copy(path, image)


def overwrite(img, image_path, jpg_path):
    if jpg_path != None:
        img.save(image_path, "PNG") # and save the new one
        os.remove(jpg_path) # remove the original image
    else:
        try:
            os.remove(image_path) # remove the original image
            img.save(image_path, "PNG") # and save the new one
        except:
            img.save(image_path, "PNG")


# uniform mode and size
def resize_and_convert(image_path, image_width = 128, image_height = 128):
    img = Image.open(image_path)

    if img.mode != "RGB":
        img = img.convert("RGBA") #convert image to "RGBA" to paste it with a white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        img.load() # required for png.split()
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        background = background.resize((image_height,image_width)) #resize to 64,64,3
        os.remove(image_path) #remove the original image
        background.save(image_path, 'PNG') #save the new image
    else:
        img = img.resize((image_height,image_width)) #resize to 64,64,3
        os.remove(image_path) #remove the original image
        img.save(image_path, 'PNG') #save the new image


def resize_and_save(image_path = None, image_width = 128, image_height = 128):
    try:
        img = Image.open(image_path)

        if img.size != (image_height,image_width):
            img = img.resize((image_height,image_width)) #resize to 64,64,3
            os.remove(image_path) #remove the original image
            img.save(image_path, 'PNG') #save the new image
    except:
        pass

def resize(img = None,image_path = None, image_width = 128, image_height = 128):
    if img == None:
        img = Image.open(image_path)
    if img.size != (image_height,image_width):
        img = img.resize((image_height,image_width)) #resize to 64,64,3
    return img

#uniform file type
def make_png(image_path):
    # removing any non png and converting it into png (jpeg, jpg)
    img = Image.open(image_path)
    if ".jpg" in image_path:
        img.save(f"{image_path[:-3]}png", "PNG")
        os.remove(image_path)
    if ".jpeg" in image_path:
        img.save(f"{image_path[:-4]}png", "PNG")
        os.remove(image_path)


#uniform the backgrounds
def make_white(img, b = 0, g = 0, r = 0):
    pixel_data = img.getdata()
    newData = []

    # check for every pixel if it is either complete black, red, green, blue
    # (the three backgrounds that occured in the dataset)
    for pixel in pixel_data:
        if item[0] == r and item[1] == g and item[2] == b: #black
            newData.append((255, 255, 255)) # append a white pixel instead
        elif item[0] != r and item[1] == g and item[2] == b: #red
            newData.append((255, 255, 255))
        elif item[0] == r and item[1] != g and item[2] == b: #green
            newData.append((255, 255, 255))
        elif item[0] == r and item[1] == g and item[2] != b: #blue
            newData.append((255, 255, 255))
        else:
            newData.append(item) #if it is none of the one keep the pixel as it is

    img.putdata(newData) # encode the pixel data whith the swapped pixels as an image
    return img

#check for white background
def uniform_background(image_path, r = 0,g = 0,b = 0):
    cv_img = cv2.imread(image_path)

    number_of_white_pix = np.sum(cv_img == (255, 255, 255))
    number_of_other_pix = np.sum(cv_img == (r, g, b))

    # if there are more black/green/red/blue pixels than white change them to white
    # the background has a very distinct color (126 for the respecitve color channel)
    # which is only used as a background because of resizing the there are also very few
    # complete black pixel (rather really dark grey but 0,0,0 is only used in the background)
    if number_of_white_pix < number_of_other_pix:
        img = Image.open(image_path)
        img2 = make_white(img) #make the background white
        os.remove(image_path) #remove and save it
        img2.save(image_path, "PNG")

def pad_to_square(img):
    # read image
    ht, wd, cc= img.shape
    # create new image of desired size and color (blue) for padding
    if ht > wd:
        ww = hh = ht
    else:
        ww = hh = wd
    color = (255,255,255)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img
    return result

def convert_pil_to_cv(img):
    # use numpy to convert the pil_image into a numpy array
    numpy_image=np.array(img)

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
    # the color is converted from RGB to BGR format
    opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image

def convert_cv_to_pil(img):
    # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that
    # the color is converted from BGR to RGB
    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_coverted)
    return pil_image

def center_focus(image_path,tol=255, border = 4):
    '''
    Removes unecessary white borders of image (makes the entire image to its focal point)
    :param tol = background color or border color to be removed
    :border = remaining border of the image
    '''

    img = cv2.imread(image_path)
    mask = img<tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    img1 = pad_to_square(img[row_start-border:row_end+border,col_start-border:col_end+border])
    # You may need to convert the color.
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(img1)
    # only resize and save if it doesn't destroy the original image
    img1 = resize(img1,image_path)

    if np.sum(convert_pil_to_cv(img1) == (255, 255, 255)) > (128*128*3)-2000:
        print(f"{path_leaf(image_path)} would have been destroyed")
        resize_and_save(convert_cv_to_pil(img),image_path)
    else:
        overwrite(img1,image_path)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def uniform(image_path, image_height = 128, image_width=128):
    try:
        # uniform the filetype
        make_png(image_path)
        try:
            # uniform the size and the transparency
            resize_and_convert(image_path,image_height,image_width)
        except:
            resize_and_save(image_path)
        # uniform the backgrounds
        uniform_background(image_path)

    except:
        pass
if __name__ == "__main__":
    #get all the needed paths
    current = os.getcwd()
    images = os.listdir("data_test")

    for image in images:
        image_path = os.path.join("data_test", image)

        uniform(image_path)
