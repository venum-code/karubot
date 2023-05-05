import nltk
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

def tokenize(sentence):
   return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

words = ["Organize", "Organizes", "Organizing"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)
