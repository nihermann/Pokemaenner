# Pokemänner
## Did you ever wondered how pokemon evolution look like? 
![](https://raw.githubusercontent.com/nihermann/Pokemaenner/main/mygif.gif)  

Check out [this](https://colab.research.google.com/github/nihermann/Pokemaenner/blob/main/Interactive_results.ipynb) notebook and make some yourself.
  
## What is Pokégan?
This Project was done as a final paper in the course "Implementing Artificial Neural Networks with Tensorflow" of the University of Osnabrueck by Michael Hüppe and Nicolai Hermann. We implemented three different versions of the popular GAN architecture to compare their performance when generating Pokémon images. Each model here being a deep convolutional GAN, a Wasserstein GAN with gradient penalty and a Auto-Encoding GAN has its own directory. The data used for this project was accumulated using the files in the preprocessing data (especially webscraping.py for gaining data and uniforming.py for data processing). In addition does every model have the needed directories required for testing the model. Moreover, does the preprocessing folder contain sample data showing the data we work with. The model architecture folder gives illustrations of the respective models. To find further information refer to our [paper](https://github.com/nihermann/Pokemaenner/blob/main/Pok%C3%A9gans_creating_new_pok%C3%A9mon_with_generative_adversarial_networks_nihermann_mhueppe.pdf). 


## Results Aegan
![](https://github.com/nihermann/Pokemaenner/blob/main/AEGAN/results/best_of_24.png)

## Results WGAN
![](https://github.com/nihermann/Pokemaenner/blob/main/WGAN/results/samples_WGAN.png)

## Naming Conventions
- for preprcessing files add preprocessing_*.py as prefix.
- images are stored in a directory called images.
- all csvs in a directory called list_data
- in general use short names but no abbreviations.


P.S The only things changed after the deadline was to add the sample results and visualize it in the README for the WGAN
