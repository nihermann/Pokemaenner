#Evan Russenberger-Rosica
#Create a Grid/Matrix of Images
import PIL, os, glob,math
from PIL import Image
from math import ceil, floor

def create_table(path, epoch = 0, padding = 0, images_per_row = 3, images_saved_per_epoch = 10, img_height = 64, process = None):

    os.chdir(path)

    images = glob.glob("*.png")
    frame_width = img_height*images_per_row
    epoch_file_start = epoch*images_saved_per_epoch
    images = images[epoch_file_start:epoch_file_start+(images_per_row*images_per_row)]          #get the first images_per_row*images_per_row images

    # if process != None:
    #     images = images[epoch_file_start:epoch_file_start+(process*process)]          #get the first images_per_row*images_per_row images

    img_width, img_height = Image.open(images[0]).size
    sf = (frame_width-(images_per_row-1)*padding)/(images_per_row*img_width)    #scaling factor

    if process != None:
        sf = (frame_width-(images_per_row-1)*padding)/(images_per_row*img_width)+0.00000001       #scaling factor
    scaled_img_width = ceil(img_width*sf)
    scaled_img_height = ceil(img_height*sf) + padding
    number_of_rows = ceil(len(images)/images_per_row)
    frame_height = ceil(sf*img_height*number_of_rows)

    new_im = Image.new('RGB', (frame_width, frame_height))

    i,j=0,0
    for num, im in enumerate(images):
        if num%images_per_row==0:
            i=0
        im = Image.open(im)
        #Here I resize my opened image, so it is no bigger than 100,100
        im.thumbnail((scaled_img_width,scaled_img_height))
        #Iterate through a 4 by 4 grid with 100 spacing, to place my image
        y_cord = (j//images_per_row)*scaled_img_height
        new_im.paste(im, (i,y_cord))
        # print(i, y_cord)
        i=(i+scaled_img_width)+padding
        j+=1

    # new_im.show()
    return new_im


PATH = r"C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\results"

def epoch_progress(path,start = 0, progress_jump_per_image = 5, table_width = 5, padding = 1, number_of_epochs = 10):
    n_epochs = progress_jump_per_image*table_width
    os.chdir(path)
    progress_name = f"progress_{start}_to_{n_epochs}"
    try:
        os.mkdir(progress_name)
    except:
        pass

    for epoch in range(start,n_epochs,progress_jump_per_image):
        print("current epoch",epoch)
        epoch_img = create_table(path, epoch = epoch)
        epoch_img.save(f"{progress_name}/{epoch}.png", "PNG", quality=100, optimize=True, progressive=True)

    progress_table = create_table(os.path.join(path,progress_name), padding = padding,  images_per_row = int(math.sqrt(table_width)), img_height = 64*3 )
    progress_table.show()
    progress_table.save(progress_name+".png", "PNG", quality=100, optimize=True, progressive=True)


epoch_progress(PATH,progress_jump_per_image = 5, table_width = 25, padding = 1)
