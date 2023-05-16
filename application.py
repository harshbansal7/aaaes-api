import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import dump, load

# !sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
# !pip install pdftotext
# !pip install nltk
# !pip install pyspellchecker
# !pip install lexicalrichness
# !pip install textstat

from collections import Counter
from spellchecker import SpellChecker
import pdftotext
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from lexicalrichness import LexicalRichness
import textstat
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.porter import PorterStemmer
snow = nltk.stem.SnowballStemmer('english')
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')
#

def extractText(pdf):
  with open(pdf, "rb") as f:
    pdf = pdftotext.PDF(f)

  pdf_content = "\n\n".join(pdf)
  pdf_content = pdf_content.replace("\n", " ")
  
  return pdf_content

def word_tokens(text):
  word_tokens = word_tokenize(text)
  return word_tokens

def sent_tokens(text):
  return sent_tokenize(text)

def lexical_richness_mtld(text):
  lex = LexicalRichness(text)
  mtld = lex.mtld()
  return mtld

def text_stats(test_data):
  print("lexical richness - ", lexical_richness_mtld(test_data))
  print("flesch_reading_ease - ", textstat.flesch_reading_ease(test_data))
  print("flesch_kincaid_grade - ", textstat.flesch_kincaid_grade(test_data))
  print("smog_index- ", textstat.smog_index(test_data))
  print("coleman_liau_index - ", textstat.coleman_liau_index(test_data))
  print("automated_readability_index - ", textstat.automated_readability_index(test_data))
  print("dale_chall_readability_score - ", textstat.dale_chall_readability_score(test_data))

def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def synonym_antonym_extractor(phrase):
     synonyms = []
     for syn in wn.synsets(phrase, pos=wn.NOUN):
          for l in syn.lemmas():
               synonyms.append(l.name())

     return set(synonyms)

def remove_words_starting_with_at(string):
    words = string.split()  # Split the string into individual words
    filtered_words = [word for word in words if not word.startswith('@')]  # Filter out words starting with '@'
    output_string = ' '.join(filtered_words)  # Join the filtered words back into a string
    return output_string

def get_inputs(text):
    inputs = []
    review = text
    review = review.lower()
    review = remove_words_starting_with_at(review)
    
    inputs.append(len(word_tokens(review)))
    inputs.append(len(sent_tokens(review)))
    inputs.append(textstat.smog_index(review))
    inputs.append(lexical_richness_mtld(review))
    inputs.append(textstat.flesch_reading_ease(review))
    inputs.append(textstat.flesch_kincaid_grade(review))

    tags = nltk.pos_tag(nltk.word_tokenize(review))
    counts = Counter(tag for word,tag in tags)

    is_noun = lambda pos: pos[:2] == 'NN'
    nouns = [word for (word, pos) in tags if is_noun(pos)] 

    inputs.append(counts['NN'])
    inputs.append(counts['VBZ'])
    inputs.append(counts['JJ'])

    all_data = []

    for i in nouns:
      syns = list(synonym_antonym_extractor(phrase=i))
      all_data.extend(syns)

    all_data.extend(nouns)

    set_1 = set(all_data)
    set_2 = set(essay_to_wordlist(review, True))

    common_words = set_1.intersection(set_2)

    inputs.append(len(common_words))
    input_params = np.array(inputs).reshape(1,-1)
    return pd.DataFrame(input_params, columns = ['word_count', 'sen_count', 'smog_index', 'lexical_richness_scores', 'flesch_reading_ease', 'flesch_kincaid_grade', 'noun_count', 'verb_count', 'adj_count', 'com_words'])


application = Flask(__name__) #Initialize the flask App
model = load('model.joblib') 

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    text = ''
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        text = extractText(uploaded_file.filename)

    inputs = get_inputs(text)
    prediction = model.predict(inputs)

    output = prediction[0]

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000, debug=True)
