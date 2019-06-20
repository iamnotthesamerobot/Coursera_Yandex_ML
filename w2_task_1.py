import re
import pandas as pd
from scipy.spatial import distance

sentences = []
words = []
new_words = []
new_sentences = []

# open & rewrite text in low register, sentence by sentence
with open('cats.txt', 'r') as file:
    for sentence in file:
        sentence = sentence.lower()
        sentences.append(sentence)
# add filtered words to a new list, sentence by sentence
for sentence in sentences:
    word = re.split('[^a-z]', sentence)
    words.append(word)
# clean up the list and pass data to new list
i = 0
for string in words:
    new_sentences.insert(i, [])
    for word in string:
        if word != '':
            new_sentences[i].append(word)
    i += 1
# create a text words dictionary with values as a serial numbers
words_dict = {}
d = 0
for sentence in new_sentences:
    for word in sentence:
        if word not in words_dict:
            words_dict[word] = d
            d += 1
# create a matrix with 0 in cells
n = len(new_sentences)
array = pd.DataFrame(words_dict, xrange(n))
for i in xrange(n):
    for j in xrange(d):
        array.iloc[i, j] = 0
# reframe the matrix by inserting the quantity of a word in a sentence in a cell
for i in xrange(n):
    for word in new_sentences[i]:
        array.loc[i, word] += 1
# calculate a cos distance form the first sentence to all other and add values to list
cos_dist = []
for i in range(n):
    cos_dist.append(distance.cosine(array.loc[0], array.loc[i]))
# create a dictionary with cos distance as a key and
cos_dist_dict = {}
for i in range(len(cos_dist)):
    cos_dist_dict[cos_dist[i]] = i
# sort data, get 2 closest values and write its values to file
answer = [str(sorted(cos_dist_dict.items())[1][1]), str(sorted(cos_dist_dict.items())[2][1])]
with open('answer_1', 'w') as output_file:
  output_file.write(' '.join(answer))
