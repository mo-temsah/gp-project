import re
from PyPDF2 import PdfReader

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)#Removes URLs
    cleanText = re.sub('RT|cc', ' ', cleanText)#Removes 'RT' (retweet) and 'cc' mentions
    cleanText = re.sub('#\S+\s', ' ', cleanText)#Removes hashtags
    cleanText = re.sub('@\S+', '  ', cleanText)#Removes @ mentions/tags
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)#Removes all special characters and punctuation
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)#Removes non-ASCII characters
    cleanText = re.sub('\s+', ' ', cleanText)#Replaces multiple spaces with a single space
    return cleanText

def pdf_to_text(file):
    if hasattr(file, 'filename'):
        if file.filename.endswith('.pdf'):
            #Handling PDF file
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif file.filename.endswith('.txt'):
            #Handling if the file is  text file
            return file.read().decode('utf-8')
    return ""

def skills_extractor(file_path):
    if file_path.endswith('.pdf'):
        #Handling PDF file
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        #Handling text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    return extract_skills_from_resume(text)

def extract_name_from_resume(text):
    name = None
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    if match:
        name = match.group()
    return name

def extract_contact_number_from_resume(text):
    contact_number = None
    pattern = r"\b(?:\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()
    return contact_number

def extract_email_from_resume(text):
    email = None
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()
    return email

def extract_skills_from_resume(text):
    with open('skills.txt', 'r') as file:
        skills_list = [line.strip() for line in file if line.strip()]
    skills = []
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)
    return skills

def extract_education_from_resume(text):
    education = []
    with open('education.txt', 'r') as file:
        education_keywords = [line.strip() for line in file if line.strip()]
    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())
    return education