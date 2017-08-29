#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'kira'

import jieba
import numpy
import pandas
from scipy.misc import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator


file = open("m.txt", 'rb')
content = file.read()
file.close()
segment = []
segs = jieba.cut(content)
for seg in segs:
    if len(seg) > 1 and seg != '\r\n':
        segment.append(seg)

words_df=pandas.DataFrame({'segment':segment})
words_df.head()

stopwords1=pandas.read_csv("stopwords.txt", index_col=False,quoting=3,sep="\t",names=['stopword'],encoding="utf8") 
stopwords2=pandas.read_csv("s.txt", index_col=False,quoting=3,sep="\t",names=['stopword'],encoding="utf8")
stopwords = stopwords1.append(stopwords2)

words_df=words_df[~words_df.segment.isin(stopwords.stopword)]
words_stat=words_df.groupby(by=['segment'])['segment'].agg({"count":numpy.size})
words_stat=words_stat.reset_index().sort_values(by="count",ascending=False) #words_stat print
print(words_stat)
#%matplotlib
#bimg=imread('rose.png')
bimg=imread('heart.jpeg')
wordcloud=WordCloud(background_color="white",mask=bimg,font_path='HYQingKongTiJ.ttf') #
word_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
word_frequence_list = {}
for key in word_frequence:
	word_frequence_list[key] = word_frequence[key]

wordcloud=wordcloud.fit_words(word_frequence_list)
bimgColors=ImageColorGenerator(bimg)
plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()
wordcloud.to_file("1.png")