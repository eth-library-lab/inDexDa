# Pew venv install
# if [ -z $(pip list | grep -F pew) ]; then
#     pip install pew
# fi
# pew new testing
# pew in testing

# PyTorch Install
# echo "Installing PyTorch if not already installed..."
# pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
# pip install torchvision

# MongoDB Install
# if [ ! (dpkg-query -W -f='${Status}' mongodb-org 2>/dev/null | grep -c "ok installed")]; then
#     echo "Installing MongoDB..."
#     wget -qO - https://www.mongodb.org/static/pgp/server-3.2.asc | sudo apt-key add -
#     echo "deb http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.2.list
#     sudo apt-get update
#     sudo apt-get install -y mongodb-org
#     sudo apt-get install build-essential python-dev
# fi

# echo "Please install the feedparser package from https://github.com/kurtmckee/feedparser"
# read -p "Is the package installed?" -n 1 -r
# echo    # (optional) move to a new line
# if [[ $REPLY =~ ^[Yy]$ ]]
# then
#     echo "Installing all package dependencies"
# fi

# REQUIRED PIP PACKAGES
# For paper scraping
pip install nltk
pip install numpy
pip install requests
pip install termcolor
pip install feedparser
pip install scikit-learn

# For NLP
pip install unidecode
pip install textsearch
pip install contractions
pip install beautifulsoup4
pip install --upgrade gensim
pip install tensorflow==1.14

# Install NLTK Libraries through python script
cat >script.py <<'NLTK_LIBRARIES'
import nltk
nltk.download()
NLTK_LIBRARIES

python script.py
rm script.py

# BERT Dependencies
pip install ktrain

