import nltk
import sys
import os
from nltk.corpus import stopwords
import math 
import collections

#English stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
en_stops = set(stopwords.words('english'))

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    #Calculate IDF values across files
    files = load_files(sys.argv[1])

    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))
    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n = FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n = SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    filenames = dict()
    path = '/Users/prashantvaidya/Downloads/questions'
    for file in os.listdir(f'{path}/{directory}'):       
        with open(f'{path}/{directory}/{file}', 'r') as f:
            filenames[f'{file}'] = f.read()
        
    return filenames


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = []
    word = ''
    for i in document:
        if i is not ' ':
            if i.isalnum():
                word += i.lower()
            elif not i.isalnum():
                continue
        else:
            if word in en_stops:
                word = ''
            else:
                if word is not '':
                    words.append(word)
                word = ''
    words.append(word)
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf = dict()
    all_words = dict()
    no_of_doc = 0

    '''Files or sentences'''
    for files in documents:
        no_of_doc += 1
        visited = set()
        for words in documents[files]:
            if words in visited:
                continue
            if words in all_words:
                all_words[words] += 1
                visited.add(words)
            else:
                all_words[words] = 1
                visited.add(words)

    for words in all_words:
        occurences = all_words[words]
        idf[words] = math.log(no_of_doc/occurences)

    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = dict()

    # Calculate tf-idf
    for file in files:
        temp = dict()
        for words in query:
            temp[words] = idfs[words]

        for words in files[file]:
            if words in temp:
                temp[words] += temp[words]

        sum = 0
        for i in temp:
            sum += temp[i]
        
        tf_idf[file] = sum
    tf_idf = sorted(tf_idf.items(), key=lambda kv: kv[1])
    tf_idf = collections.OrderedDict(tf_idf)
    
    
    top_filenames = [file for file in tf_idf]
    return top_filenames[-n:]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf = dict()
    
    # Calculate idf
    for sentence in sentences:
        sen_length = 0
        q_length = 0
        temp = dict()

        for words in sentences[sentence]:
            sen_length += 1
            if words in query:
                q_length += 1
                temp[words] = idfs[words]

        sum = 0
        for i in temp:
            sum += temp[i]

        idf[sentence] = (sum, q_length/sen_length)


    '''Change this, sort without converting to dictionary again'''

    #First sort using query term density
    idf = sorted(idf.items(), key = lambda kv: kv[1][1])
    idf = collections.OrderedDict(idf)

    #Sort using idf values
    idf = sorted(idf.items(), key = lambda kv: kv[1][0])
    idf = collections.OrderedDict(idf)

    top_sent = [sentence for sentence in idf]
    return top_sent[-n:]
 
if __name__ == "__main__":
    main()


# What are the types of supervised learning?
# When was Python 3.0 released?
# How do neurons connect in a neural network?