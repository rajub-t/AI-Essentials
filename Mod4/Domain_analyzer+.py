
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import whois
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
import os

nltk.download("punkt")
nltk.download("stopwords")


class EmailAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.domains = []
        self.usernames = []
        self.subjects = []
        self.emails = []
        self.tfidf_matrix = None
        self.feature_names = []
        self.clusters = []
        self.domain_ages = {}
        self.virustotal_scores = {}
        self.scam_indicators = []
        self.url_domains = []
        self.threat_scores = {}

    def load_data(self):
        try_encodings = ["utf-8", "latin-1", "ISO-8859-1"]
        for enc in try_encodings:
            try:
                self.df = pd.read_csv(self.file_path, encoding=enc)
                break
            except Exception:
                continue
        self.df.fillna("", inplace=True)

    def analyze_emails(self):
        for email in self.df["SenderEmail"]:
            match = re.match(r"([^@]+)@([^@]+)", email)
            if match:
                username, domain = match.groups()
                self.usernames.append(username.lower())
                self.domains.append(domain.lower())
                self.emails.append(email.lower())

    def calculate_entropy(self, items):
        counter = Counter(items)
        total = sum(counter.values())
        entropy = -sum((count / total) * math.log2(count / total) for count in counter.values())
        return entropy

    def subject_action_analysis(self):
        actions = []
        for subject in self.df["Subject"]:
            words = re.findall(r"[A-Z]{2,}", subject.upper())
            actions.extend(words)
        return Counter(actions).most_common(10)

    def extract_text_features(self):
        self.subjects = self.df["Subject"].fillna("").tolist()
        stop_words = set(stopwords.words("english"))
        cleaned_subjects = []
        for text in self.subjects:
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in stop_words]
            cleaned_subjects.append(" ".join(words))
        vectorizer = TfidfVectorizer()
        self.tfidf_matrix = vectorizer.fit_transform(cleaned_subjects)
        self.feature_names = vectorizer.get_feature_names_out()
        sums = self.tfidf_matrix.sum(axis=0)
        data = [(word, sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        return sorted(data, key=lambda x: x[1], reverse=True)[:10]

    def cluster_emails(self):
        if self.tfidf_matrix is None:
            return
        km = KMeans(n_clusters=4, random_state=42)
        self.clusters = km.fit_predict(self.tfidf_matrix)
        self.df["Cluster"] = self.clusters
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.tfidf_matrix.toarray())
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=self.clusters, cmap="viridis")
        plt.title("Email Clusters")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.colorbar()
        plt.savefig("email_clusters.png")
        plt.show()

        cluster_words = defaultdict(list)
        for i in range(4):
            indices = np.where(self.clusters == i)[0]
            words = []
            for idx in indices:
                text = self.subjects[idx].lower()
                words.extend(word_tokenize(text))
            filtered = [w for w in words if w.isalpha() and w not in stopwords.words("english")]
            cluster_words[i] = Counter(filtered).most_common(10)
        return cluster_words

    def analyze_domain_age(self):
        top_domains = [d for d, _ in Counter(self.domains).most_common(25)]
        for domain in top_domains:
            try:
                info = whois.whois(domain)
                if isinstance(info.creation_date, list):
                    creation_date = info.creation_date[0]
                else:
                    creation_date = info.creation_date
                if creation_date:
                    age = (datetime.now() - creation_date).days
                    self.domain_ages[domain] = age
            except Exception:
                self.domain_ages[domain] = None

    def domain_age_statistics(self):
        ages = [v for v in self.domain_ages.values() if v is not None]
        if not ages:
            return {}
        return {
            "min_age_days": min(ages),
            "max_age_days": max(ages),
            "avg_age_days": sum(ages) / len(ages),
        }

    def analyze_with_virustotal(self, api_key):
        try:
            from vt import Client as VTClient
        except ImportError:
            print("You must install vt-py: pip install vt-py")
            return

        client = VTClient(api_key)
        for domain in list(set(self.domains))[:20]:
            try:
                domain_obj = client.get_object(f"/domains/{domain}")
                self.virustotal_scores[domain] = domain_obj.last_analysis_stats
            except Exception:
                self.virustotal_scores[domain] = None
        client.close()

    def analyze_email_content(self):
        monetary_terms = ["usd", "dollars", "fund", "million", "payment", "cash"]
        urgency_terms = ["urgent", "immediately", "asap", "important", "now"]
        credentials_terms = ["password", "account", "login", "credentials"]
        high_risk_emails = []
        scam_categories = Counter()
        for i, row in self.df.iterrows():
            body = str(row["EmailBody"]).lower()
            score = 0
            if any(term in body for term in monetary_terms):
                score += 1
                scam_categories["Monetary Scam"] += 1
            if any(term in body for term in urgency_terms):
                score += 1
                scam_categories["Urgency"] += 1
            if any(term in body for term in credentials_terms):
                score += 1
                scam_categories["Credential Theft"] += 1
            if score >= 2:
                high_risk_emails.append((row["SenderEmail"], score))
        return high_risk_emails, scam_categories

    def analyze_urls(self):
        for urls in self.df["URLs"]:
            found = re.findall(r"https?://([\w\.-]+)", str(urls).lower())
            self.url_domains.extend(found)
        return Counter(self.url_domains).most_common(10)

    def calculate_threat_scores(self):
        for email, domain in zip(self.emails, self.domains):
            score = 0
            if self.domain_ages.get(domain, 9999) < 365:
                score += 2
            vt_data = self.virustotal_scores.get(domain, {})
            if vt_data and vt_data.get("malicious", 0) > 0:
                score += 3
            score += int(self.emails.count(email) > 1)
            self.threat_scores[email] = score

    def generate_visualizations(self, top_domains, top_emails, scam_categories, top_url_domains):
        if top_domains:
            domains, counts = zip(*top_domains)
            plt.figure(figsize=(8, 6))
            plt.pie(counts, labels=domains, autopct="%1.1f%%")
            plt.title("Top Email Domains")
            plt.savefig("top_email_domains.png")
            plt.show()

        if top_emails:
            emails, counts = zip(*top_emails)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(counts), y=list(emails))
            plt.title("Top Emails")
            plt.xlabel("Count")
            plt.ylabel("Email")
            plt.tight_layout()
            plt.savefig("top_emails.png")
            plt.show()

        if scam_categories:
            labels, values = zip(*scam_categories.items())
            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(values), y=list(labels))
            plt.title("Scam Category Frequency")
            plt.xlabel("Occurrences")
            plt.ylabel("Scam Type")
            plt.tight_layout()
            plt.savefig("scam_category_frequency.png")
            plt.show()

        if top_url_domains:
            domains, counts = zip(*top_url_domains)
            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(counts), y=list(domains))
            plt.title("Top URL Domains")
            plt.xlabel("Occurrences")
            plt.ylabel("Domain")
            plt.tight_layout()
            plt.savefig("top_url_domains.png")
            plt.show()

    def generate_report(self, top_domains, top_emails, top_terms, cluster_words, domain_stats, high_risk, scam_categories, top_url_domains):
        with open("analysis_report.txt", "w") as f:
            f.write("EMAIL SECURITY ANALYSIS REPORT\n\n")
            f.write("Top 10 Sender Domains:\n")
            for domain, count in top_domains:
                f.write(f"{domain}: {count}\n")
            f.write("\nTop 10 Sender Emails:\n")
            for email, count in top_emails:
                f.write(f"{email}: {count}\n")
            f.write("\nTop 10 Action Phrases in Subject:\n")
            for action, count in self.subject_action_analysis():
                f.write(f"{action}: {count}\n")
            f.write("\nTop 10 Text Terms:\n")
            for term, score in top_terms:
                f.write(f"{term}: {score:.4f}\n")
            f.write("\nCluster Words:\n")
            for cluster, words in cluster_words.items():
                f.write(f"Cluster {cluster}: {words}\n")
            f.write("\nDomain Age Statistics:\n")
            for k, v in domain_stats.items():
                f.write(f"{k}: {v}\n")
            f.write("\nHigh-Risk Emails:\n")
            for email, score in high_risk:
                f.write(f"{email}: risk score {score}\n")
            f.write("\nScam Categories:\n")
            for scam, count in scam_categories.items():
                f.write(f"{scam}: {count}\n")
            f.write("\nTop URL Domains:\n")
            for domain, count in top_url_domains:
                f.write(f"{domain}: {count}\n")
            f.write("\nComprehensive Threat Scores:\n")
            for email, score in self.threat_scores.items():
                f.write(f"{email}: score {score}\n")
        print("Report saved to 'analysis_report.txt'")


def main():
    file_path = input("Enter the path to your CSV file: ").strip()
    analyzer = EmailAnalyzer(file_path)
    analyzer.load_data()
    analyzer.analyze_emails()
    top_domains = Counter(analyzer.domains).most_common(10)
    top_emails = Counter(analyzer.emails).most_common(10)
    analyzer.extract_text_features()
    top_terms = analyzer.extract_text_features()
    cluster_words = analyzer.cluster_emails()
    analyzer.analyze_domain_age()
    domain_stats = analyzer.domain_age_statistics()
    api_key = input("Enter your VirusTotal API key: ").strip()
    analyzer.analyze_with_virustotal(api_key)
    high_risk, scam_categories = analyzer.analyze_email_content()
    top_url_domains = analyzer.analyze_urls()
    analyzer.calculate_threat_scores()
    analyzer.generate_visualizations(top_domains, top_emails, scam_categories, top_url_domains)
    analyzer.generate_report(top_domains, top_emails, top_terms, cluster_words, domain_stats, high_risk, scam_categories, top_url_domains)


if __name__ == "__main__":
    main()
