import glob
import os
import imageio
# pip install -q git+https://github.com/tensorflow/docs
import tensorflow_docs.vis.embed as embed


def make_gif(epoch=0, path='/content/drive/MyDrive/pokemaenner/results', list_of_files=None):
    anim_file = os.path.join(path, f"animation_of_epoch_{epoch}.gif")

    with imageio.get_writer(anim_file, mode='I') as writer:
        if list_of_files is not None:
            for image_path in list_of_files:
                image = imageio.imread(os.path.join(path, image_path))
                writer.append_data(image)
        else:
            filenames = glob.glob("*.png")
            filenames = sorted(filenames)
            for filename in filenames:
                if epoch < 9:
                    if "00" + str(epoch) in filename:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                elif epoch > 9:
                    if "0" + str(epoch) in filename:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                elif epoch > 99:
                    if str(epoch) in filename:
                        image = imageio.imread(filename)
                        writer.append_data(image)

        embed.embed_file(anim_file)


if __name__ == '__main__':
    path = r"C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\results_GP-20210402T082031Z-001\results_GP\progress_0_to_52"

    os.chdir(path)
    images = sorted(glob.glob("*.png"), key=os.path.getmtime)  # get the first images_per_row*images_per_row images
    images = [i for i in images if isinstance(i.replace(".png", ""), int)]
    make_gif(path=path, list_of_files=images)
