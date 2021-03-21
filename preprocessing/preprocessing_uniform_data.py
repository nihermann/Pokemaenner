import os
import shutil
from PIL import Image
import cv2
import numpy as np

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


#uniform file type
def make_png(image_path):
    # removing any non png and converting it into png (jpeg, jpg)
    img = Image.open(os.path.join(data_white_folder,img_path))
    if ".jpg" in image_path:
        img.save(f"{img_path[:-3]}png", "PNG")
        os.remove(os.path.join(data_white_folder,img_path))
    if ".jpeg" in image_path:
        img.save(f"{img_path[:-4]}png", "PNG")
        os.remove(os.path.join(data_white_folder,img_path))

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

# uniform mode and size
def resize_and_convert(image_path):
    img = Image.open(image_path)

    if img.mode != "RGB":
        img = img.convert("RGBA") #convert image to "RGBA" to paste it with a white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        img.load() # required for png.split()
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        background = background.resize((64,64)) #resize to 64,64,3
        os.remove(image_path) #remove the original image
        background.save(image_path, 'PNG') #save the new image

if __name__ == "__main__":
    #get all the needed paths
    current = os.getcwd()
    pokemon_folder = os.path.join(current, "new_pokemon")
    old_pokemon_folder = os.path.join(current, "old_pokemon")
    sprites_folder = os.path.join(current, "pokemon_sprites")
    alternative_folder = os.path.join(current, "pokemon_alternative_artwork")
    complete_data_folder = os.path.join(current, "complete_data", "pokemon")

    #unpacking evey folder to get one with all the images
    unpacking(pokemon_folder, complete_data)
    unpacking(old_pokemon_folder, complete_data)
    unpacking(sprites_folder, complete_data)
    unpacking(alternative_folder, complete_data)

    for image in os.listdir(complete_data_folder):
        try:
            image_path = os.path.join(complete_data_folder, image)
            # uniform the size and the transparency
            resize_and_convert(image_path)
            # uniform the filetype
            make_png(image_path)
            # uniform the backgrounds
            uniform_background(image_path)
        except:
            print(f"{img_path} removed")
            os.remove(os.path.join(data_white_folder,img_path))
            pass
