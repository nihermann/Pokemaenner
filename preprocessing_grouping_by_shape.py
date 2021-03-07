import os
import shutil
import pandas as pd
import re
from PIL import Image

def to_png():
    current = os.getcwd()
    image_data_path = os.path.join(current, "images")
    # translating jpg to png
    for folder in os.listdir(image_data_path):
        print(folder)
        for img_path in os.listdir(os.path.join(image_data_path, folder)):
            if '.jpg' in img_path:
                img_dir = os.path.join(image_data_path, folder, img_path)
                img = Image.open(img_dir)
                img.save(img_dir[:-3] + 'png', "PNG")
                os.remove(img_dir)
                print(f"Making {img_path} into PNG")
            elif '.jpeg' in img_path:
                img_dir = os.path.join(image_data_path, folder, img_path)
                img = Image.open(img_dir)
                img.save(img_dir[:-4] + 'png', "PNG")
                os.remove(img_dir)
                print(f"Making {img_path} into PNG")


def main():
    # get the paths for the images, shapes and stats data
    path_shapes = os.path.join(os.getcwd(), "csvs", "shape_by_csv.csv")
    path_stats = os.path.join(os.getcwd(), "csvs", "pokedex_(Update_05.20).csv")
    path_image_data = os.path.join(os.getcwd(), "images")

    # read the stats, shapes data, all pokemon names and image names
    stats = pd.read_csv(path_stats)
    shapes_data = pd.read_csv(path_shapes)
    pokemon_names = shapes_data['pokemon']
    images_names = os.listdir(os.path.join(path_image_data, "pokemaenner_images"))

    # turn the stats and shapes data into translatable dictionaries
    names_to_numbers = pd.Series(stats.pokedex_number.values,
                                 stats.name.values).to_dict()  # to get the pokedex numbers of a pokemon

    numbers_to_names = pd.Series(stats.name.values,  # to get the corresponding pokemon name
                                 stats.pokedex_number.values).to_dict()  # from a pokedex number

    # define some exceptions for smooth sailing
    numbers_to_names[150] = 'mewtwo'
    numbers_to_names[122] = 'mime'
    numbers_to_names[29] = 'nidoran'
    numbers_to_names[32] = 'nidoran'
    numbers_to_names[479] = 'rotom'
    numbers_to_names[555] = 'darmanitan'
    numbers_to_names[6] = 'charizard'
    numbers_to_names[658] = 'greninja'
    numbers_to_names[669] = 'flabebe'
    numbers_to_names[83] = 'farfetchd'

    names_to_shape = pd.Series(shapes_data.shapes.values,
                               shapes_data.pokemon.values).to_dict()  # to get the shape of a pokemon

    different_shapes = ["only_head", "head_and_legs", "with_fins", "insectoid_body", "quadruped_body", "multiple_wings",
                        "multiple_bodies", "tentacles", "head_and_base", "bipedal_with_tail", "bipedal_tailless",
                        "single_wings", "serpentine_body", "head_and_arms"]

    # create directories
    for shape in different_shapes:
        new_folder = os.path.join(path_image_data, shape)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

    for image in images_names:
        for pokemon in pokemon_names:
            try:
                if pokemon.lower() in image.lower():
                    shape = names_to_shape[pokemon.lower()]
                    shutil.move(os.path.join(path_image_data, "pokemaenner_images", image),
                                os.path.join(path_image_data,
                                             shape, image))
                    print(f"Moving {pokemon} to {shape}")

                elif 'Farfetch' in image:
                    shutil.move(os.path.join(path_image_data, "pokemaenner_images", image),
                                os.path.join(path_image_data,
                                             'single_wings',
                                             image))
                    break
                elif 'Sirfetch' in image:
                    shutil.move(os.path.join(path_image_data, "pokemaenner_images", image),
                                os.path.join(path_image_data,
                                             'single_wings',
                                             image))
                    break
                elif 'Flabébé' in image:
                    shutil.move(os.path.join(path_image_data, "pokemaenner_images", image),
                                os.path.join(path_image_data,
                                             'head_and_arms',
                                             image))
                    break
            except:
                print("pokemon:", image.lower(), pokemon.lower())

    for image in images_names:
        for pokedex_number in range(1000, 0, -1):
            try:
                if str(pokedex_number) in re.sub('\D', '', image):
                    pokemon = numbers_to_names[pokedex_number].lower()
                    if ' ' in pokemon:
                        if 'mega' in pokemon or 'galarian' in pokemon or 'alolan' in pokemon or 'partner' in pokemon or 'primal' in pokemon or 'white' in pokemon:
                            shape = names_to_shape[pokemon.split(' ', 1)[1]]
                        else:
                            shape = names_to_shape[pokemon.split(' ', 1)[0]]
                    elif '150' in image:
                        shape = "bipedal_with_tail"
                    else:
                        shape = names_to_shape[pokemon]

                    shutil.move(os.path.join(path_image_data, "pokemaenner_images", image),
                                os.path.join(path_image_data, shape, image))
                    print(f"{pokedex_number} to {shape}")
                    break
                elif int(re.sub('\D', '', image)) > pokedex_number:
                    break

            except:
                print(pokedex_number, image)

    # removes the directory where the files were
    if len(images_names) == 0:
        os.rmdir(images_names)


if __name__ == "__main__":
    main()
