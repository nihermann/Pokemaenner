import imageio
import time
import PIL, os, glob
from IPython import display
from postprocessing_image_process import isint
# pip install -q git+https://github.com/tensorflow/docs 
import tensorflow_docs.vis.embed as embed

def make_gif(epoch = 0, path = '/content/drive/MyDrive/pokemaenner/results', image_structure = "generated_img_*[0-9]_[0-9].png", list_of_files = None):
  anim_file = os.path.join(path, f"animation_of_epoch_{epoch}.gif")

  with imageio.get_writer(anim_file, mode='I') as writer:
    if list_of_files != None:
        for image_path in list_of_files:
            image = imageio.imread(os.path.join(path,image_path))
            writer.append_data(image)
    else:
        filenames = glob.glob("*.png")
        filenames = sorted(filenames)
        for filename in filenames:
          if epoch < 9:
            if "00"+str(epoch) in filename:
              image = imageio.imread(filename)
              writer.append_data(image)
          elif epoch > 9:
            if "0"+str(epoch) in filename:
              image = imageio.imread(filename)
              writer.append_data(image)
          elif epoch > 99:
            if str(epoch) in filename:
              image = imageio.imread(filename)
              writer.append_data(image)

    embed.embed_file(anim_file)

path = r"C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\results_GP-20210402T082031Z-001\results_GP\progress_0_to_52"

os.chdir(path)
images = sorted(glob.glob("*.png"), key=os.path.getmtime)          #get the first images_per_row*images_per_row images
images = [i for i in images if isint(i.replace(".png", ""))]
make_gif(path = path, list_of_files = images)
