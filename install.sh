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


# Dataset Extraction Dependencies
# pip install PyPDF2
