from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import csv
from nltk.corpus import stopwords
stopw  = set(stopwords.words('english'))
import re
from PyPDF2 import PdfReader

#Load dataset:
jd_df=pd.read_csv(r'jd_cleaned_data.csv')

#Read skills from CSV file
file_path=r'skills.csv'
with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    skills_list = [skill for row in csv_reader for skill in row]

skills = []
#Function to extract skills from text
def extract_skills(text):
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills

#Function to extract text from PDF
def extract_text_from_pdf(file):
     reader = PdfReader(file)
     text = ''
     for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
     return text

def skills_extractor(file_path):
        resume_text = extract_text_from_pdf(file_path)

        # Extract skills from resume text
        skills = list(extract_skills(resume_text))
        return skills

#Load the extracted  resume skills:
file_path=r'CV.pdf'
skills=[]
skills.append(' '.join(word for word in skills_extractor(file_path)))

def ngrams(string, n=3):
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(skills)

nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
jd_test = (jd_df['Processed_JD'].values.astype('U'))

def getNearestN(query):
  queryTFIDF_ = vectorizer.transform(query)
  distances, indices = nbrs.kneighbors(queryTFIDF_)
  return distances, indices

distances, indices = getNearestN(jd_test)
test = list(jd_test) 
matches = []

for i,j in enumerate(indices):
    dist=round(distances[i][0],2)
  
    temp = [dist]
    matches.append(temp)
    
matches = pd.DataFrame(matches, columns=['Match confidence'])

#Following recommends Top 5 Jobs based on the candidate resume:
jd_df['match']=matches['Match confidence']
jd_df.head(5).sort_values('match')