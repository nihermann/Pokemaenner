import shutil
import os


if __name__ == '__main__':
    rootdir = os.path.dirname(os.path.abspath("top_level_file.txt"))

    for subdir in os.listdir(rootdir):
        subpath = os.path.join(rootdir,subdir)
        try:
            file_names = os.listdir(subpath)
            for file_name in file_names:
                shutil.move(os.path.join(subpath, file_name), rootdir)
        except:
            pass

        try:
            os.rmdir(subpath)
        except OSError as e:
            print("Error: %s : %s" % (subpath, e.strerror))
