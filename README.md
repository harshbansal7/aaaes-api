# AAAES - Automated Assignment Assessment and Evaluation System
The Automated Assignment Assessment and Evaluation System (AAAES) is a versatile tool that 
simplifies the often-tedious process of grading student assignments. By leveraging the latest 
advances in machine learning and natural language processing, the system can accurately and 
objectively evaluate student work, saving educators valuable time and effort. The system is 
designed to handle a wide range of Assignment types, from coding projects to written essays. By 
streamlining the grading process, it empowers teachers to spend more time providing feedback 
and mentoring their students, rather than spending countless hours on checking codes or grading 
papers.

## Current Components of this System
1. A **Frontend** form created using basic HTML, CSS to send inputs to the backend in a seamless way.
2. The **Backend** created using FLASK. Its an API which has two endpoints- `/` for form page @ home, and `/predict` for `@predict` where all the text processing takes place. 

## What exactly happens when you submit the form with a ZIP / PDF File for Evaluation?
1. The File Format is checked first. If its a ZIP, PDF Files are extracted *temporarily*.
2. For text in every file, the following parameters are found out using the subsequent libraries:-
    - `File Name` - using API request
    - `Word Count` - Number of Tokens - using [NLTK](https://pypi.org/project/nltk/) (`nltk.tokenize.word_tokenize`)
    - `Sentence Count` - Number of Sentences - using [NLTK](https://pypi.org/project/nltk/) (`nltk.tokenize.sent_tokenize`)
    - `Smog Index` -  Measure of how many years of education the average person needs to have to understand a text - using [TextStat Package](https://pypi.org/project/textstat/) (`textstat.smog_index`)
    - `Lexical Richness Score` - Refers to the range and variety of vocabulary deployed in a text by a speaker/writer - using [lexicalrichness](https://pypi.org/project/lexicalrichness/) package
    - `Flesch Reading Ease` - Helpful to assess the ease of readability in a document ([More Info](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease)) - using [TextStat Package](https://pypi.org/project/textstat/)
    - `Flesch Kincaid Grade` -  This is a grade formula in that a score of 9.3 means that a ninth grader would be able to read the document - using [TextStat Package](https://pypi.org/project/textstat/)
    - `Noun Count` - Number of Nouns in the given text - using using [NLTK](https://pypi.org/project/nltk/)
    - `Verb Count` - Number of Verbs in the given text - using using [NLTK](https://pypi.org/project/nltk/)
    - `Adjective Count` - Number of Adjectives in the given text - using using [NLTK](https://pypi.org/project/nltk/)
    - `Common Word Count` - Number of Common Words in the given text - using using [NLTK](https://pypi.org/project/nltk/)

    For every file submitted using the form, this data is logged into a Pandas DF, following which this is passed into a Model to produce a score. The Model has been discussed in the next Section.

3. The Model produces a score b/w 1-10 for the input parameters of one file. The Score is stored into the Pandas DF, in a score column for corresponding file. 
4. As per requirement, the scores are normalized so that they fall in b/w some min and max value if opted in the form. 

## Machine Learning Part
### Dataset
The Dataset is a widely used dataset for this purpose, the [ASAP-AES dataset by The Hewlett Foundation](https://www.kaggle.com/c/asap-aes) given as a part of a competition on Kaggle. 
**At the end of the data-preprocessing part (not discussed in this description), we had 12976 Scored Essays, scored on a scale of 1-10.** This was split into Training and Testing, 10381 and 2595 entries respectively. 

### The Classifier Model
The Model is a `RandomForestClassifier` Model implemented using `Sci-kit Learn Library` with 100 estimators and all default parameters. 

### Evaluation of the Model
**This Model Scored `0.615` which was found out using the `cohen_kappa_score` method of the `sklearn.metrics` library.**

A metric `Quadratic weighted kappa` was used to assess the model's performance. Kappa or Cohen’s Kappa is like classification accuracy, except that it is normalized at the baseline of random chance on your dataset: It basically tells you how much better your classifier is performing over the performance of a classifier that simply guesses at random according to the frequency of each class. [Read More at Kaggle](https://www.kaggle.com/code/reighns/understanding-the-quadratic-weighted-kappa). 

---

## What can be added to this system?
The System only checks assignment based on their **Readibility Only**. Before it can be deployed for real world tasks, integration of **Semantic Analysis** and **Plagarism Checks** will be something that can be added. 

Still it is a good tool for teachers and other educators to reduce their workload by a little, which can be huge if the amount of assignments that need checking is enormous. 

---

Created with ❤️ by Harsh Bansal


