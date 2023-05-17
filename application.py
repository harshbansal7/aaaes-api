from typing import final
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, make_response
import pickle
from joblib import dump, load
import os
from scipy.stats import percentileofscore
from zipfile import ZipFile
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

@application.route('/predict', methods=['POST'])
def predict():
    formdata = request.form
    directory_path = os.path.join(os.getcwd(), 'pdfs')

    file = request.files['file']

    result_df_2 = pd.DataFrame(columns=['filename', 'word_count', 'sen_count', 'smog_index', 'lexical_richness_scores', 'flesch_reading_ease', 'flesch_kincaid_grade', 'noun_count', 'verb_count', 'adj_count', 'com_words'])
    final_df = pd.DataFrame(columns=['File Name', 'Base Score (10)'])

    def process_pdf_file(pdf_file):
        nonlocal result_df_2  # Declare the variables as nonlocal
        pdf_path = os.path.join(directory_path, pdf_file)
        input_params = get_inputs(extractText(pdf_path))  # Call your function with the PDF file path
        score = model.predict(input_params)
        input_params['filename'] = pdf_file
        input_params['score'] = score
        result_df_2 = pd.concat([result_df_2, input_params], ignore_index=True)

    def extract_pdf_files_from_zip(zip_file):
        with ZipFile(zip_file, 'r') as zip:
            zip.extractall(directory_path)
        pdf_files = [file for file in os.listdir(directory_path) if file.endswith(".pdf")]
        for pdf_file in pdf_files:
            process_pdf_file(pdf_file)

        nonlocal final_df
        final_df['File Name'] = result_df_2['filename']
        final_df['Base Score (10)'] = result_df_2['score']

        if(formdata['normalize-score'] == 'yes'):
            final_df['Actual Percentile'] = round(final_df['Base Score (10)'].apply(lambda x: percentileofscore(final_df['Base Score (10)'], x)), 2)
            if(formdata['custom-max-min-option'] == 'yes'):
                min_percentile = int(formdata['min-score'])
                max_percentile = int(formdata['max-score'])
            else:
                min_percentile = 50
                max_percentile = 100
            final_df['Normalized Percentile'] = round(min_percentile + ((final_df['Actual Percentile'] - final_df['Actual Percentile'].min()) / (final_df['Actual Percentile'].max() - final_df['Actual Percentile'].min())) * (max_percentile - min_percentile), 2)
        
        if(formdata['round-to-int'] == 'yes'):
            final_df = final_df.apply(lambda x: np.round(x) if np.issubdtype(x.dtype, np.number) else x)

        final_df = final_df.sort_values(by=['Base Score (10)'], ascending=False)


    # Check if the file is a ZIP file or a PDF file
    if file.filename.endswith('.zip'):
        # Assuming 'directory_path' is the desired extraction path
        extract_pdf_files_from_zip(file)
    elif file.filename.endswith('.pdf'):
        process_pdf_file(file.filename)
    else:
        # Handle unsupported file format
        print("Unsupported file format.")

    # Continue with the rest of your code using 'result_df' and 'result_df_2'

    csv_string = final_df.to_csv(index=False)
    response = make_response(csv_string)
    response.headers['Content-Disposition'] = 'attachment; filename=data.csv'
    response.headers['Content-type'] = 'text/csv'

    return response

    # return render_template('index.html', prediction_text='Here is the updated Result', result_table=final_df.to_html())

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000, debug=True)
