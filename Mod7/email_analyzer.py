# email_analyzer.py
import os
import csv
import logging
import sqlite3
from datetime import datetime
from flask import Flask, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import whois
import joblib
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import re
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)

# Flask app setup
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///email_analysis.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORT_FOLDER'] = 'reports'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Define DB Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    avg_threat_score = db.Column(db.Float)
    details = db.Column(db.Text)

with app.app_context():
    db.create_all()

# Load classifier and vectorizer
classifier = joblib.load('pretrained_classifier.joblib')
vectorizer = joblib.load('pretrained_vectorizer.joblib')

# Scam keyword patterns
scam_keywords = ["lottery", "prize", "urgent", "click here", "verify", "bank", "password", "account"]

# Email analysis utilities
def extract_domain(email):
    return tldextract.extract(email).registered_domain

def compute_entropy(domain):
    prob = [float(domain.count(c)) / len(domain) for c in dict.fromkeys(list(domain))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            age_days = (datetime.now() - creation_date).days
            return age_days
    except:
        pass
    return -1

def virustotal_score(domain):
    # Stub: Assume neutral threat score of 0.5
    return 0.5

def analyze_content(subject, body):
    content = f"{subject} {body}".lower()
    score = 0
    for word in scam_keywords:
        if word in content:
            score += 1
    urls = re.findall(r'https?://\S+', content)
    score += len(urls) * 0.5
    if "password" in content or "verify" in content:
        score += 2
    return score / max(len(scam_keywords) + 2, 1)

def predict_with_classifier(text):
    x = vectorizer.transform([text])
    prediction = classifier.predict(x)[0]
    return 1 if prediction in ['spam', 'bad'] else 0

def analyze_email(row):
    domain = extract_domain(row['sender_email'])
    domain_entropy = compute_entropy(domain)
    domain_age = get_domain_age(domain)
    vt_score = virustotal_score(domain)
    content_score = analyze_content(row['subject'], row['body'])
    prediction_score = predict_with_classifier(f"{row['subject']} {row['body']}")

    domain_age_score = 1 if domain_age < 30 and domain_age != -1 else 0.5

    final_score = np.mean([domain_entropy / 5.0, vt_score, content_score, domain_age_score, prediction_score])
    return final_score

def cluster_emails(df):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df['subject'] + ' ' + df['body'])
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())
    df['cluster'] = labels
    df['pca_x'] = reduced[:, 0]
    df['pca_y'] = reduced[:, 1]
    return df

def adversarial_attack_stub():
    pass

# Routes
@app.route('/')
def home():
    return "<h2>Welcome to Email Analyzer</h2><p><a href='/login'>Login</a> or <a href='/register'>Register</a></p>"

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        if User.query.filter_by(username=username).first():
            return "User already exists"
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return "<form method='POST'>Username: <input name='username'><br>Password: <input name='password' type='password'><br><input type='submit'></form>"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user'] = username
            return redirect(url_for('upload'))
        return "Invalid credentials"
    return "<form method='POST'>Username: <input name='username'><br>Password: <input name='password' type='password'><br><input type='submit'></form>"

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        f = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(filepath)
        df = pd.read_csv(filepath)
        df['threat_score'] = df.apply(analyze_email, axis=1)
        df = cluster_emails(df)
        avg_score = df['threat_score'].mean()

        report_path = os.path.join(app.config['REPORT_FOLDER'], f.filename + '_report.txt')
        df.to_csv(report_path, index=False)

        new_report = Report(filename=f.filename, avg_threat_score=avg_score, details=df.to_csv(index=False))
        db.session.add(new_report)
        db.session.commit()

        return f"Report generated. Average Threat Score: {avg_score:.2f}<br><a href='/reports'>View Reports</a>"

    return "<form method='POST' enctype='multipart/form-data'><input type='file' name='file'><input type='submit'></form>"

@app.route('/reports')
def reports():
    if 'user' not in session:
        return redirect(url_for('login'))
    reports = Report.query.all()
    html = "<h2>Past Reports</h2>"
    for report in reports:
        html += f"<p><b>{report.filename}</b> - {report.timestamp} - Avg Score: {report.avg_threat_score:.2f}</p>"
    return html

if __name__ == '__main__':
    app.run(debug=True)