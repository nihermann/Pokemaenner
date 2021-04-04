import os
import glob
import math
from math import ceil
from PIL import Image, ImageDraw


def create_table(path, epoch=0, padding=0, images_per_row=3, images_saved_per_epoch=10, img_height=64, process_till=0):
    # '''creates table with images of one epoch'''
    os.chdir(path)

    images = sorted(glob.glob("*.png"))
    frame_width = img_height * images_per_row
    epoch_file_start = epoch * images_saved_per_epoch
    images = images[epoch_file_start:epoch_file_start + (
                images_per_row * images_per_row)]  # get the first images_per_row*images_per_row images

    if images_saved_per_epoch != 10:
        images = images[epoch_file_start:epoch_file_start + (math.sqrt(images_saved_per_epoch) * math.sqrt(
            images_saved_per_epoch))]  # get the first images_per_row*images_per_row images

    if process_till >= 1:
        images = sorted(glob.glob("*.png"), key=os.path.getmtime)  # get the first images_per_row*images_per_row images
        images = [i for i in images if isinstance(i.replace(".png", ""), int)]

    img_width, img_height = Image.open(images[0]).size
    sf = (frame_width - (images_per_row - 1) * padding) / (images_per_row * img_width)  # scaling factor

    scaled_img_width = ceil(img_width * sf)
    scaled_img_height = ceil(img_height * sf) + padding
    number_of_rows = ceil(len(images) / images_per_row)
    frame_height = ceil(sf * img_height * number_of_rows)

    new_im = Image.new('RGB', (frame_width, frame_height))

    i, j = 0, 0
    for num, im in enumerate(images):
        if num % images_per_row == 0:
            i = 0
        im = Image.open(im)
        # Here I resize my opened image, so it is no bigger than 100,100
        im.thumbnail((scaled_img_width, scaled_img_height))
        # Iterate through a 4 by 4 grid with 100 spacing, to place my image
        y_cord = (j // images_per_row) * scaled_img_height
        new_im.paste(im, (i, y_cord))
        # print(i, y_cord)
        i = (i + scaled_img_width) + padding
        j += 1

    # new_im.show()
    return new_im


def epoch_counter(epoch, color=(255, 255, 255), img_size=20):
    # '''make an image with the number of the current epoch'''
    img_width = len(str(epoch)) * 10
    img = Image.new('RGB', (img_width, 10), color=color)
    d = ImageDraw.Draw(img)
    d.text((3, 0), str(epoch), fill=(0, 0, 0))
    img = img.resize((int(img_width * (img_size / img_size)), img_size))
    return img


def mark_with_epoch(img, epoch, img_size=10):
    # '''paste the current epoch onto the image'''
    epoch_number = epoch_counter(epoch, img_size=10)
    img.paste(epoch_number, (0, 0))
    return img


def epoch_progress(path, start=0, progress_jump_per_image=5, n_epochs=45, epochs_per_row=5, padding=1,
                   number_of_epochs=10, images_per_row_epoch=3):
    # '''create table with multiple epochs each visualizing images from that specific epoch'''
    os.chdir(path)
    progress_name = f"progress_{start}_to_{n_epochs}"
    try:
        os.mkdir(progress_name)
    except:
        pass

    for epoch in range(start, n_epochs, progress_jump_per_image):
        # for each epoch create a table with its images
        print("current epoch", epoch)
        epoch_img = create_table(path, epoch=epoch, images_per_row=images_per_row_epoch)
        epoch_img = mark_with_epoch(epoch_img, epoch, img_size=30)  # mark the whole image with the current epoch
        epoch_img.save(f"{progress_name}/{epoch}.png", "PNG", quality=100, optimize=True,
                       progressive=True)  # save the image

    # create_table a table with every epoch in it
    progress_table = create_table(os.path.join(path, progress_name), padding=padding, images_per_row=epochs_per_row,
                                  img_height=64 * images_per_row_epoch,
                                  process_till=int(n_epochs / progress_jump_per_image))
    progress_table.show()
    progress_table.save(progress_name + ".png", "PNG", quality=100, optimize=True, progressive=True)


if __name__ == '__main__':
    PATH = r"C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\results_GP-20210402T082031Z-001\results_GP"
    epoch_progress(PATH, progress_jump_per_image=1, epochs_per_row=4, padding=1, n_epochs=52)
