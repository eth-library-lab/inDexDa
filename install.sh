# Pew venv install
pip install pew
pew new index
pew in index

# PyTorch Install
# echo "Installing PyTorch if not already installed..."
# pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
# pip install torchvision

# MongoDB Install
if [ ! (dpkg-query -W -f='${Status}' mongodb-org 2>/dev/null | grep -c "ok installed")]; then
    echo "Installing MongoDB..."
    wget -qO - https://www.mongodb.org/static/pgp/server-3.2.asc | sudo apt-key add -
    echo "deb http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.2.list
    sudo apt-get update
    sudo apt-get install -y mongodb-org
    sudo apt-get install build-essential python-dev
fi

echo "Please install the feedparser package from https://github.com/kurtmckee/feedparser"
read -p "Is the package installed?" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing all package dependencies"
fi

# Required Packages
if [ -z $(pip list | grep -F numpy) ]; then
    pip install numpy
fi
if [ -z $(pip list | grep -F requests) ]; then
    pip install requests
fi
if [ -z $(pip list | grep -F scikit-learn) ]; then
    pip install scikit-learn
fi
if [ -z $(pip list | grep -F gensim) ]; then
    pip install --upgrade gensim
fi
if [ -z $(pip list | grep -F nltk) ]; then
    pip install nltk
fi

# For paper scraping
pip install feedparser
pip install pymongo
pip install requests
pip install termcolor

# For NLP
# pip install inflect
pip install unidecode
pip install textsearch
pip install contractions
pip install beautifulsoup4
# pip install wordcloud
# pip install matplotlib
pip install gensim

# BERT Dependencies
# pip install pandas
# pip install ipywidgets
# pip install apex
# pip install pytorch-transformers
# pip install tensorboardX

# For processing pipeline
