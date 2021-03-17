import requests
from bs4 import BeautifulSoup
import os

#get the url with teh sprites
url = 'https://pokemondb.net/sprites'

# request it
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')

# find all the attributed tags (links that contain more information aka lead to
# a differnet site here: the pokemon) from the website
sprites_urls = []
for a in soup.find_all('a', href=True):
    # of all the attributed tags get only the ones that are pokemon with their sprites
    if '/sprites/' in  a['href'] :
        # append this tag with the base link to form the complete link to the pokemon
        sprites_urls.append('https://pokemondb.net'+ a['href'])


def imagedown(url, folder):
    try:
        new_folder = os.path.join(os.getcwd(),folder)
        #making a direcotry by joining the current one with the given one
        os.mkdir(new_folder)
    except:
        pass

    #change to the given folder and request the given url and parse its html
    os.chdir(new_folder)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    # find all images in the given parsed html file (the one with all the sprites of the respective pokemon)
    images = soup.find_all('img')

    # for every image there is for that pokemon make the filename valid and
    # and download it
    for image in images:
        name = image['alt']
        link = image['src']
        with open(name.replace(' ', '_').replace('/', '').replace('&', '') + '.jpeg' , 'wb') as f:
            im = requests.get(link)
            f.write(im.content)
            print('Writing: ', name)

    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

# for every pokemon found in the first file find every possible sprite for it
# and make a new folder with its name
for pokemon in sprites_urls[:2]:
    imagedown(pokemon, pokemon.replace('https://pokemondb.net/sprites/', ''))
