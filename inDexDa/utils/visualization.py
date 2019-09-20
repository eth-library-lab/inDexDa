from wordcloud import WordCloud
import matplotlib.pyplot as plt


def viz(corpus, max_words):
    wordcloud = WordCloud(background_color='black',
                          # colormap='Oranges',
                          collocations=False,
                          width=1200,
                          height=1000,
                          normalize_plurals=True,
                          relative_scaling=0.5,
                          max_words=max_words,
                          max_font_size=100).generate(str(corpus))

    plt.figure('Word Cloud')
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
