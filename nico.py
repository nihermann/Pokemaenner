from PIL import Image
import imagehash
import os
hash0 = imagehash.average_hash(Image.open(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/480_static_55656.png'))
hash1 = imagehash.average_hash(Image.open(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/483_static_13520.png'))
cutoff = 5


alist, blist, clist, dlist, elist, flist, glist, hlist, ilist = ([] for i in range(9))

def get_hash(path):
    return imagehash.average_hash(Image.open(path))

a =  get_hash(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/480_static_5790.png')
b =  get_hash(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/481_static_34668.png')
c =  get_hash(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/481_static_64048.png')
d =  get_hash(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/481_static_83724.png')
e =  get_hash(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/481_static_94504.png')
f =  get_hash(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/482_static_9636.png')
g =  get_hash(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/482_static_59932.png')
h =  get_hash(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/484_static_4812.png')
i =  get_hash(r'C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download/485_static_33372.png')

classifier_list = [a,b,c,d,e,f,g,h,i]
def dissimilarity(a,b):
    return a - b


image_path = r"C:\Users\michi\Osnabrueck\3_Semester\Pokemaenner\drive_download"
for img in os.listdir(image_path):
    img_path = os.path.join(image_path, img)
    hash = get_hash(img_path)
    dissimilarities = [dissimilarity(hash,c) for c in classifier_list]
    index = dissimilarities.index(min(dissimilarities))
    os.rename(img_path, img_path.replace("_static_", f"_{index}_static_"))
