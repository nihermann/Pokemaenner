import requests
from bs4 import BeautifulSoup
import os
from preprocessing_uniform_data import uniform, center_focus, resize_and_save
import io
import PIL
from PIL import Image


# gets all the pokemon sites of their list
def get_pokemon(url, identifier="pokedex"):
    soup = get_soup(url)
    # find all the attributed tags (links that contain more information aka lead to
    # a different site here: the pokemon) from the website
    pokemon_urls = []
    for a in soup.find_all('a', href=True):
        # of all the attributed tags get only the ones that are pokemon (identifiable by the url (pokedex/sprites))
        if identifier in a['href']:
            # append this tag with the base link to form the complete link to the pokemon
            pokemon_urls.append('https://pokemondb.net' + a['href'])
    all_pokemon = list(set(pokemon_urls))
    all_pokemon.sort()
    return all_pokemon


def listed_sprites(soup):
    # find all images in the given parsed html file
    lists = soup.find_all('span')
    listed_pokemon = []
    # get all the lists
    for list in lists:
        # get all the href in each list
        listed_pokemon += [image['href'] for image in list.find_all('a', href=True)]

    return listed_pokemon


def listed_artwork(soup):
    originals = [image['src'] for image in soup.find_all('img')]
    alternatives = [image['href'] for image in soup.find_all('a', href=True) if 'artwork' in image['href']]
    return originals + alternatives


def get_soup(url):
    # request the given url
    r = requests.get(url)
    # parse it into a soup
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


def download(images):
    # for every image in the list make the filename valid and
    # and download it
    for image in images:
        name = image.replace('https://img.pokemondb.net/', '').replace(' ', '_').replace('-', '_').replace('/', '_').replace('&', '_')

        if '_back_' not in name and '.gif' not in name:
            with open(name, 'wb') as f:
                im = requests.get(image)
                f.write(im.content)
            uniform(os.path.join(os.getcwd(), name), 128, 128)


def imagedown(url, folder):
    folder = os.path.join(os.getcwd(), folder)
    # making a directory by joining the current one with the given one
    os.makedirs(folder, exist_ok=True)
    # change to the given folder and request the given url and parse its html
    os.chdir(folder)

    # get the soup of the pokemon for both its sprites and its artworks
    # sprites_soup = get_soup(url.replace('pokedex', 'sprites'))
    artwork_soup = get_soup(url.replace('pokedex', 'artwork'))

    # find all images in the given parsed html file
    # sprites = listed_sprites(sprites_soup)
    artworks = listed_artwork(artwork_soup)

    # download all the images in both of the lists
    # download(sprites)
    download(artworks)
    # move to the original path
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)


def remove_corrupted_images(path):
    for file in os.listdir(path):
        try:
            with open(os.path.join(path, file), 'rb') as f:
                img = Image.open(io.BytesIO(f.read()))
        except PIL.UnidentifiedImageError as e:
            os.remove(os.path.join(path, file))
            print(f"Removed {file} because it is corrupted.")


if __name__ == "__main__":
    # for every pokemon found in the first site (the national pokedex) find every possible sprite for it
    # and make a new folder with its name (or not)
    url = 'https://pokemondb.net/pokedex/national'
    count = 0
    for pokemon in get_pokemon(url)[1:]:
        try:
            count = count + 1
            imagedown(pokemon, folder="data")
            print(
                f"Number of downloaded pokemon: {count}/903. Current pokemon: {pokemon.replace('https://pokemondb.net/pokedex/', '')}",
                sep='', end="\r", flush=True)
        except:
            # move to the original path
            path_parent = os.path.dirname(os.getcwd())
            os.chdir(path_parent)
    print()  # clear \r from above print.
    images = os.listdir("data")

    # sometimes do to laggy internet it doesn't resize properly only do this if
    # if you can see that there are wrongly sized images in your data folder (checks before resizing but still)
    for image in images:
        if "artwork" in image:
            resize_and_save(os.path.join(os.getcwd(), "data", image), 128, 128)

    for image in images:
        image_path = os.path.join(os.getcwd(),"data", image)
        try:
            if "artwork" not in image and "sprite" in image:
                center_focus(image_path)
        except Exception as e:
            print("Failed to center: ", image, "with exception", e)

    remove_corrupted_images("data")