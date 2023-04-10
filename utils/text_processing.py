import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def process_sentence(
    s,
    lower=True,
    remove_number=True,
    remove_punctuation=True,
    tokenize=False,
    remove_stop=False,
    stem=False,
    lemmatize=False,
):
    """
    Process a sentence.
    :param s: the sentence
    :type s: str
    :param lower: whether lower all characters. e.g. 'A' -> 'a'
    :type lower: bool
    :param remove_number: whether remove all numbers
    :type remove_number: bool
    :param remove_punctuation: whether remove all punctuations
    :type remove_punctuation: bool
    :param tokenize: whether tokenize the sentence. if true, will return a list of words rather than str.
    :type tokenize: bool
    :param remove_stop: whether remove the stop words
    :type remove_stop: bool
    :param stem: whether apply stemming. e.g. "studies" -> "studi"
    :type stem: bool
    :param lemmatize: whether apply lemmatize. e.g. "studies" -> "study"
    :type lemmatize: bool
    :return: the processed sentence.
    :rtype: str if not tokenize else list
    """
    s = s.strip()
    s = s.replace("\t", " ")
    s = s.replace("\n", " ")
    if lower:
        s = s.lower()
    if remove_number:
        s = re.sub(r"\d +", "", s)
    if remove_punctuation:
        s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())

    if tokenize or remove_stop or stem or lemmatize:
        s = word_tokenize(s)
        if remove_stop:
            s = [w for w in s if w not in stopwords.words("english")]
        if stem:
            stemmer = PorterStemmer()
            s = [stemmer.stem(w) for w in s]
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            s = [lemmatizer.lemmatize(w) for w in s]
        if not tokenize:
            s = " ".join(s)

    return s


def process_text(text, **kwargs):
    """
    Process a text (i.e. a list of sentences)
    :param text: a list of sentences
    :type text: list
    :param kwargs: the parameters to the process_sentence function
    :return: the processed text
    :rtype: list
    """
    return [process_sentence(s, **kwargs) for s in text]
