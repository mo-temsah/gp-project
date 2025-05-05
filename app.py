from flask import Flask, request, render_template, redirect, url_for, session
from flask_mysqldb import MySQL
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PyPDF2 import PdfReader
import csv
import numpy as np
from ftfy import fix_text
import re
import MySQLdb.cursors
from utils import (
    cleanResume,
    pdf_to_text,
    extract_name_from_resume,
    extract_contact_number_from_resume,
    extract_email_from_resume,
    extract_skills_from_resume,
    extract_education_from_resume
)

from job_recommender import (
    ngrams,
    getNearestN,
    jd_df,
    extract_skills,
    extract_text_from_pdf,
    skills_extractor
)

app = Flask(__name__)

#conecting to database 
app.secret_key = 'temsah'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'login_system'

mysql = MySQL(app)

#Loading models 
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))
job_recommender_model = pickle.load(open('models/job_recommender_model.pkl', 'rb'))
job_recommender_vectorizer = pickle.load(open('models/job_recommender_vectorizer.pkl', 'rb'))

def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job

#Function to process the resume and recommend jobs
def process_resume(file_path):
    resume_skills = skills_extractor(file_path)
    resume_text = pdf_to_text(file_path)
    processed_text = f"{resume_text} {' '.join(resume_skills)}"
    processed_text = cleanResume(processed_text)

    #Get AI predictions and calculate matches
    resume_features = job_recommender_vectorizer.transform([processed_text])
    resume_score = job_recommender_model.predict(resume_features)[0]
    
    #Ensure resume_score is between 0 and 1
    resume_score = max(0, min(1, float(resume_score)))
    
    job_texts = jd_df['Processed_JD'] + ' ' + \
                jd_df['Industry'] + ' ' + jd_df['Sector']
    job_texts = job_texts.apply(cleanResume)
    job_vectors = job_recommender_vectorizer.transform(job_texts)
    content_similarities = cosine_similarity(resume_features, job_vectors)[0]
    
    #Calculate skill matches from job description
    skill_match = jd_df['Processed_JD'].apply(
        lambda x: len(set(resume_skills) & set(extract_skills_from_resume(str(x)))) / \
        (len(set(extract_skills_from_resume(str(x)))) + 1)
    )

    #Get industry and sector diversity
    industry_penalty = jd_df['Industry'].map(jd_df['Industry'].value_counts(normalize=True))
    sector_penalty = jd_df['Sector'].map(jd_df['Sector'].value_counts(normalize=True))
    
    #Combine scores with diversity penalties
    jd_df['match'] = (
        (resume_score * content_similarities * 40) +  # Model prediction weighted by content similarity
        (skill_match * 40) +  # Skills match
        ((1 - industry_penalty) * 10) +  # Industry diversity
        ((1 - sector_penalty) * 10)  # Sector diversity
    )
    
    #Normalize matches to 0-100 scale
    max_match = jd_df['match'].max()
    jd_df['match'] = (jd_df['match'] / max_match * 100).clip(0, 100).round(1)
    
    #Get top 10 diverse recommendations
    recommendations = jd_df.sort_values('match', ascending=False)
    
    #Remove duplicate job listings based on Job Title and Company Name
    recommendations = recommendations.drop_duplicates(subset=['Job Title', 'Company Name'])
    
    diverse_recommendations = []
    seen_industries = set()
    
    for _, job in recommendations.iterrows():
        if len(diverse_recommendations) >= 10:  
            break
        if job['Industry'] not in seen_industries or len(seen_industries) >= 5:
            #After 5 unique industries allowing duplicates to fill remaining spots
            diverse_recommendations.append(job)
            seen_industries.add(job['Industry'])
    
    result_df = pd.DataFrame(diverse_recommendations)
    
    #Add job URLs for each recommendation
    result_df['url'] = result_df.apply(
        lambda x: url_for('job_posting', 
                         job_title=x['Job Title'],
                         company=x['Company Name']), axis=1)
    
    return result_df[['Job Title', 'Company Name', 'Location', 'Industry', 
                     'Sector', 'Average Salary', 'match', 'url']]

def rank_resumes(job_description, resumes):
    #Preprocess job description and resumes
    job_description = cleanResume(job_description)
    processed_resumes = [cleanResume(resume) for resume in resumes]
    
    #Extract key requirements and skills from job description
    job_skills = extract_skills_from_resume(job_description)
    
    #Enhanced vectorizer with better feature engineering
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.9,
        analyzer=ngrams,
        lowercase=True,
        stop_words='english',
        max_features=10000,
        ngram_range=(1, 3)
    )
    
    #Create document vectors
    documents = [job_description] + processed_resumes
    vectors = vectorizer.fit_transform(documents).toarray()  # Convert to dense array
    
    #Calculate content similarity - fixing vector shapes
    job_vector = vectors[0].reshape(1, -1)  # Reshape to 2D array
    resume_vectors = vectors[1:]
    content_similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    
    #Calculate skill-based similarity
    skill_scores = []
    for resume in resumes:
        resume_skills = extract_skills_from_resume(resume)
        skill_match = len(set(job_skills) & set(resume_skills)) / len(job_skills) if job_skills else 0
        skill_scores.append(skill_match)
    
    #Calculate education relevance
    education_scores = []
    for resume in resumes:
        edu = extract_education_from_resume(resume)
        edu_score = 0.8 if edu else 0.4
        education_scores.append(edu_score)
    
    #Combine all scores with weights and convert to percentage
    final_scores = (
        0.5 * content_similarities +  # 50% weight for content matching
        0.3 * np.array(skill_scores) +  # 30% weight for skills matching
        0.2 * np.array(education_scores)  # 20% weight for education
    ) * 100  # Convert to percentage
    
    return final_scores

@app.route('/')
def index():
    session.clear()
    return render_template('welcome.html')

@app.route('/resume')
def resume_page():
    if 'username' not in session: #checks if the user logged in as jobseeker
        return redirect(url_for('login'))
    if session['user_type'] != 'job_finder':
        return redirect(url_for('login'))
    return render_template('resume.html')

@app.route('/pred', methods=['POST']) 
def pred():
    if 'resume' in request.files: #checks if a resume got uploaded 
        file = request.files['resume'] 
        filename = file.filename
        detailed_recommendations = None
        
        if filename.endswith('.pdf'): #checks if it's pdf file uploaded
            temp_path = 'temp_resume.pdf'
            file.save(temp_path)
            text = pdf_to_text(file) #changing the file format from pdf to text to analayise the text 
            try:
                detailed_recommendations = process_resume(temp_path)
            except Exception as e:
                print(f"Error in process_resume: {str(e)}")
                detailed_recommendations = None
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
            try:
                # Save txt content temporarily for processing
                temp_path = 'temp_resume.txt'
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                detailed_recommendations = process_resume(temp_path)
            except Exception as e:
                print(f"Error in process_resume: {str(e)}")
                detailed_recommendations = None
        else:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")

        try:
            predicted_category = predict_category(text)
            recommended_job = job_recommendation(text)
            phone = extract_contact_number_from_resume(text)
            email = extract_email_from_resume(text)
            extracted_skills = extract_skills_from_resume(text)
            extracted_education = extract_education_from_resume(text)
            name = extract_name_from_resume(text)

            recommendations_dict = None
            if detailed_recommendations is not None:
                recommendations_dict = detailed_recommendations.to_dict('records')

            return render_template('resume.html',
                                predicted_category=predicted_category,
                                recommended_job=recommended_job,
                                phone=phone, 
                                name=name, 
                                email=email,
                                extracted_skills=extracted_skills,
                                extracted_education=extracted_education,
                                detailed_recommendations=recommendations_dict)
        except Exception as e:
            print(f"Error processing resume: {str(e)}")
            return render_template('resume.html', message="Error processing resume. Please try again.")
            
    else:
        return render_template("resume.html", message="No resume file uploaded.")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_type = request.form.get('user_type', 'job_finder')
        
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s AND user_type = %s', 
                      (username, password, user_type))
        user = cursor.fetchone()
        cursor.close()
        
        if user:
            session['username'] = username
            session['user_type'] = user_type
            
            # Use a dictionary to map user types to their respective dashboards
            dashboard_routes = {
                'hr': 'hr_dashboard',
                'job_finder': 'resume_page'
            }
            return redirect(url_for(dashboard_routes.get(user_type, 'login')))
        
        return render_template('login.html', message='Invalid credentials')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user_type = request.form.get('user_type', 'job_finder')
        
        cursor = mysql.connection.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        if cursor.fetchone():
            cursor.close()
            return render_template('register.html', message='Username already exists')
        
        cursor.execute('INSERT INTO users (username, email, password, user_type) VALUES (%s, %s, %s, %s)',
                      (username, email, password, user_type))
        mysql.connection.commit()
        cursor.close()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/hr_dashboard')
def hr_dashboard():
    if 'username' not in session or session['user_type'] != 'hr':
        return redirect(url_for('login'))
    return render_template('hr_dashboard.html')

@app.route('/rank_resumes_endpoint', methods=['POST'])
def rank_resumes_endpoint():
    if 'username' not in session or session['user_type'] != 'hr':
        return redirect(url_for('login'))

    job_description = request.form.get('job_description', '')
    if not job_description:
        return render_template('hr_dashboard.html', message="Please provide a job description")

    if 'resumes' not in request.files:
        return render_template('hr_dashboard.html', message="Please upload resume files")

    files = request.files.getlist('resumes')
    resumes = []
    filenames = []

    for file in files:
        if file.filename.endswith('.pdf'):
            text = pdf_to_text(file)
            resumes.append(text)
            filenames.append(file.filename)
        elif file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
            resumes.append(text)
            filenames.append(file.filename)

    if not resumes:
        return render_template('hr_dashboard.html', message="No valid PDF or TXT files uploaded")

    scores = rank_resumes(job_description, resumes)
    
    #Create results list with filename and score pairs
    results = []
    for filename, score in zip(filenames, scores):
        results.append({
            'filename': filename,
            'score': round(float(score), 2)  #Round to 2 decimal places
        })
    
    # Sort results by score in descending order
    results.sort(key=lambda x: x['score'], reverse=True)

    return render_template('hr_dashboard.html', 
                         results=results,  
                         job_description=job_description)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/job_posting/<job_title>/<company>')
def job_posting(job_title, company):
    # Find the job details from the dataset
    job = jd_df[(jd_df['Job Title'] == job_title) & (jd_df['Company Name'] == company)].iloc[0]
    
    return render_template('job_posting.html', 
                         job_title=job_title,
                         company=company,
                         location=job['Location'],
                         industry=job['Industry'],
                         sector=job['Sector'],
                         salary=job['Average Salary'],
                         description=job['Job Description'] if 'Job Description' in job else 'Job description not available')


if __name__ == '__main__':
    app.run(debug=True)

