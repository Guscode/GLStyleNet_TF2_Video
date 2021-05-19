#!/usr/bin/env bash

VENVNAME=stylevenv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# problems when installing from requirements.txt
brew install wget
pip install ipython
pip install jupyter
pip install matplotlib
pip install opencv-python
pip install tensorflow

URL=https://github.com/EndyWon/GLStyleNet/releases/download/v1.0/vgg19.pkl.bz2
FILE=./vgg19.pkl.bz2
wget $URL -O $FILE

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt


echo "build $VENVNAME"