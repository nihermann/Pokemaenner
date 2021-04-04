# Pokemänner
![](https://raw.githubusercontent.com/nihermann/Pokemaenner/main/WGAN/results/generated_img_152_0.png)
## Naming Conventions
- for preprcessing files add preprocessing_*.py as prefix.
- images are stored in a directory called images.
- all csvs in a directory called csvs
- in general use short names but no abbreviations.


#How to commit for michu
commit - snapshot of the whole project at certain time
commit saves its predecssor changes in the files author time commit messages
identified by hash 

version control: what changes for different versions : remeber what you did and where you went wrong 
git commit 

the repository: contains all commits -> saves all old files in hidden folder 

file status:
stages: file will be commite with the next commit
modifies: file is registered for git and was changed since last commit
undmodified: file is registered in git but equal to the last commit
untracked git knows the file exists but won't do anything with it 

git add: staged so in the next commit it will will be comiited 
git commit: back to undmodified 
after editing add again so it will be commited in the next time
git commit -a commits all the modified files 
git status: shows you the current status 

git workflow:
 git clone repository link
 git status 
 git pull
 edit
 git add
 git commit
 git push
