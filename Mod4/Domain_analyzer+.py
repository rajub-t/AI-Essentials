import pandas as pd
import numpy as np
# import matplotlib # No specific backend setting, let Matplotlib choose
# matplotlib.use('Agg') # REMOVED to allow interactive plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import re
import whois # requires `pip install python-whois`
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
# `requests` and `time` will be imported conditionally within analyze_with_virustotal

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download("punkt", quiet=False) # Set quiet=False for visibility
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK 'stopwords'...")
    nltk.download("stopwords", quiet=False) # Set quiet=False for visibility


class EmailAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.domains = []
        self.usernames = []
        self.subjects_text_list = []
        self.emails = []
        self.tfidf_matrix = None
        self.feature_names = []
        self.clusters = [] # This will store cluster labels for each email
        self.domain_ages = {}
        self.virustotal_scores = {} # Stores domain: {stats}
        self.domain_creation_dates = {}

    def load_data(self):
        try_encodings = ["utf-8", "latin-1", "ISO-8859-1"]
        for enc in try_encodings:
            try:
                self.df = pd.read_csv(self.file_path, encoding=enc)
                print(f"Successfully loaded {self.file_path} with encoding {enc}.")
                break
            except Exception:
                continue
        if self.df is None:
            print(f"Error: Could not load or decode {self.file_path}.")
            exit()
        self.df.fillna("", inplace=True) # Fill NaN with empty strings
        # Ensure 'Subject' column exists before trying to access it.
        if "Subject" in self.df.columns:
            self.subjects_text_list = self.df["Subject"].fillna("").tolist()
        else:
            print("Warning: 'Subject' column not found in CSV. Subject-related analyses will be affected.")
            self.subjects_text_list = [] # Ensure it's an empty list

        # Ensure 'SenderEmail' column exists
        if "SenderEmail" not in self.df.columns:
            print("Error: 'SenderEmail' column not found in CSV. This column is crucial for analysis.")
            # Create an empty column to prevent KeyErrors, though results will be minimal
            self.df["SenderEmail"] = "" 
            # Consider exiting if SenderEmail is absolutely critical for all downstream tasks
            # exit() 


    def analyze_emails_and_usernames(self):
        if "SenderEmail" not in self.df.columns or self.df["SenderEmail"].empty:
            print("Skipping email and username analysis as 'SenderEmail' column is missing or empty.")
            return

        for email_str_original in self.df["SenderEmail"]:
            email_str = str(email_str_original).strip() 

            if not email_str: 
                continue

            self.emails.append(email_str.lower())

            match = re.match(r"([^@]+)@([^@]+)", email_str)
            if match:
                username, domain_part = match.groups()
                # Basic validation for domain part (must contain at least one dot and not be just dots)
                if username.strip() and domain_part.strip() and '.' in domain_part and not domain_part.strip().replace('.', '').isalnum() == False :
                    self.usernames.append(username.lower())
                    self.domains.append(domain_part.lower())


    def get_username_patterns(self):
        if not self.usernames:
            return 0, 0, 0.0, 0 
        
        num_usernames = len(self.usernames)
        usernames_with_numbers = sum(1 for u in self.usernames if re.search(r'\d', u))
        usernames_with_underscores = sum(1 for u in self.usernames if '_' in u)
        avg_username_length = sum(len(u) for u in self.usernames) / num_usernames if num_usernames > 0 else 0.0
        return usernames_with_numbers, usernames_with_underscores, avg_username_length, num_usernames


    def calculate_entropy(self, items):
        if not items:
            return 0.0
        counter = Counter(items)
        total = sum(counter.values())
        if total == 0:
            return 0.0
        entropy = -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)
        return entropy

    def subject_action_analysis(self):
        action_keywords = ['urgent', 'reply', 'attention', 'important', 'asap', 
                           'respond', 'immediate', 'confirm', 'response needed', 'update']
        
        keyword_email_counts = Counter()
        subjects_with_any_action_indices = set()

        all_subjects = self.subjects_text_list
        total_subjects = len(all_subjects)
        if total_subjects == 0:
            return [], 0, 0

        for i, subject_text in enumerate(all_subjects):
            subject_lower = str(subject_text).lower() # Ensure string conversion
            action_found_in_this_subject = False
            for keyword in action_keywords:
                if keyword in subject_lower: 
                    keyword_email_counts[keyword] += 1
                    action_found_in_this_subject = True
            if action_found_in_this_subject:
                subjects_with_any_action_indices.add(i)
        
        top_phrases_details = []
        sorted_keyword_counts = sorted(keyword_email_counts.items(), key=lambda item: item[1], reverse=True)

        for keyword, count in sorted_keyword_counts[:10]:
             percentage = (count / total_subjects) * 100 if total_subjects > 0 else 0
             top_phrases_details.append((keyword, count, percentage))
        
        num_emails_with_actions = len(subjects_with_any_action_indices)
        return top_phrases_details, num_emails_with_actions, total_subjects


    def extract_text_features(self):
        print("Preparing text for analysis...")
        if not self.subjects_text_list:
            print("No subjects found for text feature extraction.")
            self.tfidf_matrix = None
            self.feature_names = []
            return []

        print(f"Prepared {len(self.subjects_text_list)} text samples for analysis")
        
        stop_words = set(stopwords.words("english"))
        cleaned_subjects = []
        for text in self.subjects_text_list:
            words = word_tokenize(str(text).lower()) # Ensure string conversion
            words = [w for w in words if w.isalpha() and w not in stop_words]
            cleaned_subjects.append(" ".join(words))
        
        if not any(cleaned_subjects): # Check if all subjects became empty after cleaning
            print("All subjects are empty after preprocessing (e.g., only stopwords or non-alphabetic characters). TF-IDF cannot be computed.")
            self.tfidf_matrix = None
            self.feature_names = []
            return []

        print("Applying TF-IDF vectorization...")
        vectorizer = TfidfVectorizer(max_features=1000)
        try:
            self.tfidf_matrix = vectorizer.fit_transform(cleaned_subjects)
            self.feature_names = vectorizer.get_feature_names_out()
            print(f"Successfully extracted {self.tfidf_matrix.shape[1]} text features from {self.tfidf_matrix.shape[0]} subjects.")
        except ValueError as e:
             print(f"TF-IDF Vectorization error: {e}. This can happen if all documents are empty after preprocessing.")
             self.tfidf_matrix = None
             self.feature_names = []
             return []

        if self.tfidf_matrix is None or self.tfidf_matrix.shape[1] == 0: # Check if features were extracted
            print("No features extracted by TF-IDF. Skipping top terms calculation.")
            return []

        sums = self.tfidf_matrix.sum(axis=0)
        data = []
        for i, term in enumerate(self.feature_names):
            data.append((term, sums[0, i]))
            
        return sorted(data, key=lambda x: x[1], reverse=True)[:10]

    def cluster_emails(self): 
        print("Applying K-means clustering...")
        pca_data_for_plotting = None 
        if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            print("TF-IDF matrix not available or empty. Skipping clustering.")
            return {}, pca_data_for_plotting

        n_samples_for_clustering = self.tfidf_matrix.shape[0]
        # K-means n_clusters must be <= n_samples. Target 4 clusters if possible.
        n_clusters_to_use = min(4, n_samples_for_clustering) 

        if n_samples_for_clustering < 2 or n_clusters_to_use < 2 : 
            print(f"Not enough samples ({n_samples_for_clustering}) or too few distinct groups possible for meaningful clustering (target clusters: {n_clusters_to_use}). Skipping clustering.")
            return {}, pca_data_for_plotting


        km = KMeans(n_clusters=n_clusters_to_use, random_state=42, n_init='auto')
        self.clusters = km.fit_predict(self.tfidf_matrix)
        
        # df index must align with subjects_text_list if this is to work.
        # This assumes tfidf_matrix rows correspond directly to df rows.
        if len(self.df) == len(self.subjects_text_list) and len(self.subjects_text_list) == len(self.clusters):
            # Create a temporary Series for clusters, aligned with df's index
            # This is safer if df has been filtered or re-indexed.
            # However, if subjects_text_list was derived from an already filtered df, direct assignment might be okay.
            # Simplest assumption: subjects_text_list directly corresponds to df rows used for TFIDF
             self.df["Cluster"] = self.clusters # This line can cause issues if lengths don't match due to prior filtering.
        elif len(self.df) == len(self.clusters): # If clusters align with original df rows
             self.df["Cluster"] = self.clusters
        else:
            print(f"Warning: DataFrame length ({len(self.df)}) or subjects_text_list length ({len(self.subjects_text_list)}) does not match cluster results length ({len(self.clusters)}). Cannot add 'Cluster' column to DF reliably.")


        print("Cluster distribution:")
        cluster_counts = Counter(self.clusters)
        total_emails_in_clusters = len(self.clusters)
        for i in range(n_clusters_to_use):
            count = cluster_counts.get(i,0) 
            percentage = (count / total_emails_in_clusters) * 100 if total_emails_in_clusters > 0 else 0
            print(f"  Cluster {i}: {count} emails ({percentage:.1f}%)")

        print("\nMost common words in each cluster:")
        cluster_words_map = defaultdict(list)
        
        for i in range(n_clusters_to_use):
            print(f"Cluster {i} top terms:")
            indices = np.where(self.clusters == i)[0]
            if len(indices) == 0:
                print("  (No emails in this cluster)")
                cluster_words_map[i] = []
                continue

            words_in_cluster = []
            for idx in indices:
                if idx < len(self.subjects_text_list): # self.subjects_text_list was used for TF-IDF
                    text = str(self.subjects_text_list[idx]).lower() 
                    words_in_cluster.extend(re.findall(r'\b\w+\b', text)) 
            
            stop_words_set = set(stopwords.words("english"))
            filtered_cluster_words = [w for w in words_in_cluster if w.isalpha() and w not in stop_words_set]
            
            top_10_words = [word for word, count in Counter(filtered_cluster_words).most_common(10)]
            cluster_words_map[i] = top_10_words
            for word in top_10_words:
                print(f"  {word}")
        
        print("\nDomain distribution by cluster:")
        # Check if 'Cluster' column was successfully added and 'SenderEmail' exists
        if "Cluster" in self.df.columns and "SenderEmail" in self.df.columns and pd.api.types.is_numeric_dtype(self.df['Cluster']):
            for i in range(n_clusters_to_use):
                print(f"Cluster {i} top domains:")
                cluster_df = self.df[(self.df["Cluster"] == i) & (self.df["SenderEmail"].astype(str).str.strip() != "")]
                if cluster_df.empty:
                    print("  (No emails in this cluster with valid SenderEmail for domain analysis)")
                    continue

                cluster_email_domains = []
                for email_str in cluster_df["SenderEmail"]:
                    email_s = str(email_str).strip().lower()
                    if '@' in email_s:
                        parts = email_s.split('@')
                        if len(parts) == 2 and parts[1].strip() and '.' in parts[1].strip(): 
                             cluster_email_domains.append(parts[1].strip())
                
                if not cluster_email_domains:
                    print("  (No valid domains found in this cluster's emails)")
                    continue

                domain_counts_in_cluster = Counter(cluster_email_domains)
                total_domains_in_cluster_count = len(cluster_email_domains)
                
                for domain, count in domain_counts_in_cluster.most_common(3):
                    percentage = (count / total_domains_in_cluster_count) * 100 if total_domains_in_cluster_count > 0 else 0
                    print(f"  {domain}: {count} emails ({percentage:.1f}%)")
        else:
            print("  Skipping domain distribution by cluster: 'Cluster' (numeric) or 'SenderEmail' column issue, or clustering incomplete.")

        if self.tfidf_matrix is not None and self.tfidf_matrix.shape[0] >= 2 : 
            print("\n--- Preparing data for VISUALIZING EMAIL CLUSTERS (PCA) ---")
            print("Reducing dimensions with PCA...")
            n_components_pca = min(2, self.tfidf_matrix.shape[0]-1, self.tfidf_matrix.shape[1]-1) # n_comp < min(n_samp, n_feat)
            n_components_pca = max(1, n_components_pca) # Must be at least 1

            if n_components_pca < 1 or self.tfidf_matrix.shape[0] <= n_components_pca or self.tfidf_matrix.shape[1] <= n_components_pca: # Stricter check
                 print(f"Cannot perform PCA. Not enough samples ({self.tfidf_matrix.shape[0]}) or features ({self.tfidf_matrix.shape[1]}) for {n_components_pca} PCA components. Skipping PCA.")
            else:
                pca = PCA(n_components=n_components_pca, random_state=42)
                try:
                    # Ensure TF-IDF matrix is not excessively sparse leading to all-zero rows/cols if that's an issue for PCA
                    # This is usually handled by TfidfVectorizer itself (e.g. min_df) or by data having content.
                    reduced_features = pca.fit_transform(self.tfidf_matrix.toarray()) # .toarray() can be memory intensive for large matrices
                    pca_data_for_plotting = (reduced_features, self.clusters) 
                    print(f"PCA data prepared with {n_components_pca} component(s). Visualization will be generated.")
                except Exception as e:
                    print(f"Error during PCA: {e}")
        else:
            print("\n--- VISUALIZING EMAIL CLUSTERS ---") 
            print("Skipping PCA visualization due to insufficient data or TF-IDF error.")

        return cluster_words_map, pca_data_for_plotting


    def analyze_domain_age(self):
        print("Analyzing age for top 25 unique domains only...")
        valid_domains_for_age = [d for d in self.domains if d and d.strip() and '.' in d]
        unique_domain_counts = Counter(valid_domains_for_age) 
        top_domains_list = [d for d, _ in unique_domain_counts.most_common(25) if d and d.strip()]
        
        if not top_domains_list:
            print("No valid domains found to analyze age.")
            return

        print(f"Analyzing age for {len(top_domains_list)} unique top domains...")
        
        for domain in top_domains_list:
            try:
                # Stricter regex for a valid domain name for WHOIS
                if not re.match(r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$", domain):
                    print(f"Skipping invalid domain format for WHOIS: {domain}")
                    self.domain_ages[domain] = None
                    self.domain_creation_dates[domain] = None
                    continue

                info = whois.whois(domain)
                creation_date_obj = None 
                if info.creation_date: 
                    # Handle cases where creation_date might be a list
                    current_date = info.creation_date
                    if isinstance(current_date, list):
                        # Prefer datetimes, then sort and take earliest if multiple datetimes
                        datetime_dates = sorted([d for d in current_date if isinstance(d, datetime)])
                        if datetime_dates:
                            creation_date_obj = datetime_dates[0]
                    elif isinstance(current_date, datetime):
                        creation_date_obj = current_date
                
                if creation_date_obj and isinstance(creation_date_obj, datetime):
                    age_days = (datetime.now() - creation_date_obj).days
                    if age_days < 0 : # Creation date in future, likely parsing error or bad data
                        print(f"Warning: Domain {domain} has a creation date in the future ({creation_date_obj.strftime('%Y-%m-%d')}). Skipping age calculation.")
                        self.domain_ages[domain] = None
                        self.domain_creation_dates[domain] = None
                    else:
                        self.domain_ages[domain] = age_days
                        self.domain_creation_dates[domain] = creation_date_obj 
                        print(f"Domain {domain}: {age_days} days old (created: {creation_date_obj.strftime('%Y-%m-%d')})")
                else:
                    # print(f"No valid creation date found or parsed for {domain}. Date info from WHOIS: {info.creation_date}")
                    self.domain_ages[domain] = None
                    self.domain_creation_dates[domain] = None
            except whois.parser.PywhoisError as e: 
                print(f"WHOIS parsing error for {domain}: {e}. Domain might be unsupported or WHOIS data malformed.")
                self.domain_ages[domain] = None
                self.domain_creation_dates[domain] = None
            except Exception as e: 
                print(f"Could not get WHOIS info for {domain}: {type(e).__name__} - {e}") 
                self.domain_ages[domain] = None
                self.domain_creation_dates[domain] = None

    def get_domain_age_statistics(self):
        ages = [v for v in self.domain_ages.values() if v is not None and isinstance(v, (int, float)) and v >= 0]
        if not ages:
            return {"avg_age_days": 0.0, "min_age_days": 0, "max_age_days": 0, "count": 0}
        
        return {
            "avg_age_days": sum(ages) / len(ages) if ages else 0.0,
            "min_age_days": min(ages) if ages else 0,
            "max_age_days": max(ages) if ages else 0,
            "count": len(ages)
        }

    def analyze_with_virustotal(self, api_key):
        if not api_key:
            print("VirusTotal API key not provided. Skipping VirusTotal analysis.")
            self.virustotal_scores = {}
            return
        try:
            import requests
            import time
        except ImportError:
            print("Error: 'requests' library not found. Please install it (`pip install requests`) to use VirusTotal analysis.")
            self.virustotal_scores = {}
            return

        print("Analyzing domains with VirusTotal (top 10 unique, valid domains)...")
        # Ensure domains are somewhat valid before sending to VT
        valid_domains_for_vt = [d for d in self.domains if d and d.strip() and '.' in d and re.match(r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}$", d)]
        unique_domains_to_check = list(Counter(valid_domains_for_vt).keys())[:10] 

        if not unique_domains_to_check:
            print("No valid domains found for VirusTotal analysis after filtering.")
            self.virustotal_scores = {}
            return

        for domain in unique_domains_to_check:
            url = f"https://www.virustotal.com/api/v3/domains/{domain}"
            headers = {"x-apikey": api_key, "accept": "application/json"}
            try:
                response = requests.get(url, headers=headers, timeout=20) 
                response.raise_for_status() 
                data = response.json()
                
                stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                if stats: 
                    self.virustotal_scores[domain] = {
                        'malicious': stats.get('malicious', 0),
                        'suspicious': stats.get('suspicious', 0),
                        'undetected': stats.get('undetected', 0),
                        'harmless': stats.get('harmless', 0),
                        'timeout': stats.get('timeout', 0)
                    }
                else: # Domain might be in VT but no analysis_stats, or not in VT at all (404 handled below)
                    print(f"  VirusTotal: No analysis stats found for {domain}. It might be new or data incomplete.")
                    self.virustotal_scores[domain] = {'error': 'no_analysis_stats'}

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code == 401:
                    print(f"  Error with VirusTotal for {domain}: Unauthorized (401 - check API key).")
                    self.virustotal_scores[domain] = {'error': 'unauthorized_401'}
                elif status_code == 404: 
                    print(f"  VirusTotal: Domain {domain} not found in VT database (404).")
                    self.virustotal_scores[domain] = {'error': 'not_found_404'}
                elif status_code == 429:
                    print(f"  Error with VirusTotal for {domain}: Rate limit exceeded (429). Try again later.")
                    self.virustotal_scores[domain] = {'error': 'rate_limit_exceeded_429'}
                else:
                    print(f"  HTTP error {status_code} with VirusTotal for {domain}: {e}")
                    self.virustotal_scores[domain] = {'error': f'http_error_{status_code}'}
            except requests.exceptions.Timeout:
                print(f"  Timeout error with VirusTotal for {domain}.")
                self.virustotal_scores[domain] = {'error': 'timeout'}
            except requests.exceptions.RequestException as e: # Other network/request issues
                print(f"  Request error with VirusTotal for {domain}: {type(e).__name__} - {e}")
                self.virustotal_scores[domain] = {'error': f'request_exception_{type(e).__name__}'}
            except Exception as e: 
                print(f"  Unexpected error processing VirusTotal for {domain}: {type(e).__name__} - {e}")
                self.virustotal_scores[domain] = {'error': f'unexpected_{type(e).__name__}'}
            
            time.sleep(1.5) # VT Public API has strict rate limits (e.g. 4/min). Be respectful.

    def identify_high_risk_emails_from_csv(self):
        high_risk_emails = []
        if self.df is None or self.df.empty:
            print("DataFrame not loaded. Cannot identify high-risk emails from CSV.")
            return high_risk_emails

        required_cols = ['Label', 'SenderEmail']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            print(f"Required column(s) {', '.join(missing_cols)} not found in CSV for high-risk email identification.")
            return high_risk_emails
        
        try:
            # Ensure 'Label' can be compared to 1. Coerce to numeric, non-convertible become NaN.
            # Fill NaN with a value that won't match 1 (e.g., -1) to avoid errors/misidentification.
            labels_numeric = pd.to_numeric(self.df['Label'], errors='coerce').fillna(-1).astype(int)
            
            # Filter rows where numeric Label is 1 and SenderEmail is not empty/whitespace
            high_risk_df = self.df[
                (labels_numeric == 1) & 
                (self.df['SenderEmail'].astype(str).str.strip() != "")
            ]

            for _, row in high_risk_df.iterrows():
                email = str(row['SenderEmail']).strip()
                original_label_val = row['Label'] # Use original label for the reason string
                if email: 
                        high_risk_emails.append((email, f"Identified as High Risk (Label={original_label_val})"))
        except Exception as e:
            print(f"Error processing 'Label' or 'SenderEmail' column for high-risk email identification: {e}")
        
        return high_risk_emails


    def generate_report_file(self, top_domains_list, top_emails_list, top_terms_list, 
                           cluster_words_output, domain_age_stats_dict, 
                           virustotal_results, high_risk_emails_from_csv,
                           scam_categories_counter, top_url_domains_list, comprehensive_threat_scores):
        report_filename = "analysis_report.txt"
        print(f"Generating {report_filename}...")
        try:
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write("Domain and Email Analysis Report\n")
                f.write("=" * 30 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Analyzed file: {self.file_path}\n")
                total_rows_in_csv = len(self.df) if self.df is not None else 0
                f.write(f"Total rows in CSV: {total_rows_in_csv}\n")
                f.write(f"Total emails parsed from 'SenderEmail' (used for domain/username extraction): {len(self.emails)}\n")

                f.write("\n\n=== DOMAIN ANALYSIS ===\n")
                if top_domains_list:
                    f.write("Top 10 sender domains (from valid parsed domains):\n")
                    for i, (domain, count) in enumerate(top_domains_list, 1):
                        f.write(f"  {i}. {domain}: {count} emails\n")
                else:
                    f.write("  No valid domain data to display for top list (e.g., 'SenderEmail' empty or all malformed).\n")
                
                # Use self.domains which contains all parsed domains for these stats
                valid_domains_for_report = [d for d in self.domains if d and d.strip() and '.' in d]
                num_unique_domains_report = len(set(valid_domains_for_report))
                domain_entropy_report = self.calculate_entropy(valid_domains_for_report)
                max_possible_entropy_report = math.log2(num_unique_domains_report) if num_unique_domains_report > 0 else 0
                f.write(f"\nTotal unique valid domains found: {num_unique_domains_report}\n")
                f.write(f"Domain distribution entropy: {domain_entropy_report:.2f}")
                if num_unique_domains_report > 0 :
                    f.write(f" (max possible for these domains: {max_possible_entropy_report:.2f})\n")
                else:
                    f.write("\n")


                f.write("\n\n=== EMAIL ADDRESS ANALYSIS ===\n")
                if top_emails_list:
                    f.write("Top 10 sender email addresses (from valid parsed emails):\n")
                    for i, (email, count) in enumerate(top_emails_list, 1):
                        f.write(f"  {i}. {email}: {count} emails\n")
                else:
                    f.write("  No valid email address data to display for top list.\n")
                
                valid_sender_emails_report = [e for e in self.emails if e and e.strip() and '@' in e and e.split('@')[-1].strip() and '.' in e.split('@')[-1].strip()]
                num_unique_emails_report = len(set(valid_sender_emails_report))
                f.write(f"\nTotal unique valid email addresses found: {num_unique_emails_report}\n")

                nums, underscores, avg_len, total_usernames = self.get_username_patterns()
                if total_usernames > 0:
                    f.write("\nEmail address username patterns (from valid parsed emails):\n")
                    f.write(f"  Usernames analyzed: {total_usernames}\n")
                    f.write(f"  Usernames containing numbers: {nums} ({nums/total_usernames*100:.1f}%)\n")
                    f.write(f"  Usernames containing underscores: {underscores} ({underscores/total_usernames*100:.1f}%)\n")
                    f.write(f"  Average username length: {avg_len:.1f} characters\n")
                else:
                    f.write("\n  No usernames to analyze for patterns (e.g., all emails malformed or 'SenderEmail' empty).\n")

                f.write("\n\n=== SUBJECT ACTION ANALYSIS ===\n")
                action_phrases_data, emails_with_actions, total_subjects_analyzed = self.subject_action_analysis() # Recalculate for report consistency
                if total_subjects_analyzed > 0:
                    perc_emails_with_actions = (emails_with_actions / total_subjects_analyzed) * 100
                    f.write(f"Found {emails_with_actions} emails ({perc_emails_with_actions:.1f}%) with action-oriented phrases in {total_subjects_analyzed} subjects analyzed.\n")
                else:
                    f.write("No subjects available or analyzed for action phrases (e.g., 'Subject' column missing or empty).\n")

                if action_phrases_data:
                    f.write("\nTop action phrases found in subject lines (up to 10):\n")
                    for i, (phrase, count, percentage) in enumerate(action_phrases_data, 1):
                        f.write(f"  {i}. '{phrase}': {count} emails ({percentage:.1f}% of analyzed subjects)\n")
                elif total_subjects_analyzed > 0: # Analyzed but no keywords found
                    f.write("  No pre-defined action phrases identified in subjects.\n")

                f.write("\n\n=== TEXT FEATURE EXTRACTION (TF-IDF from Subjects) ===\n")
                if top_terms_list: # This list comes from main() after extract_text_features()
                    f.write(f"Top terms across all email subjects (TF-IDF, up to 10 from {self.tfidf_matrix.shape[0] if self.tfidf_matrix is not None else 0} subjects):\n")
                    for term, score in top_terms_list:
                        f.write(f"  {term}: (score) {score:.2f}\n")
                else:
                    f.write("  No TF-IDF terms to display (e.g., no subjects, all subjects empty after preprocessing, or TF-IDF error).\n")
                
                f.write("\n\n=== EMAIL CLUSTERING (K-Means on Subject TF-IDF) ===\n")
                if self.clusters is not None and len(self.clusters) > 0: 
                    n_clusters_report = len(set(self.clusters))
                    f.write(f"Clustering performed into {n_clusters_report} groups (from {len(self.clusters)} subjects).\n")
                    f.write("Cluster distribution:\n")
                    cluster_counts_report = Counter(self.clusters) 
                    total_emails_in_clusters_report = len(self.clusters)
                    for i in sorted(cluster_counts_report.keys()): 
                        count = cluster_counts_report[i]
                        percentage = (count / total_emails_in_clusters_report) * 100 if total_emails_in_clusters_report > 0 else 0
                        f.write(f"  Cluster {i}: {count} emails ({percentage:.1f}%)\n")
                    
                    if cluster_words_output: # This comes from main() after cluster_emails()
                        f.write("\nMost common words in each cluster (from subjects):\n")
                        for cluster_id in sorted(cluster_words_output.keys()):
                            words = cluster_words_output[cluster_id]
                            f.write(f"  Cluster {cluster_id}: {', '.join(words) if words else '(No prominent words or no emails in this cluster)'}\n")
                    
                    if "Cluster" in self.df.columns and "SenderEmail" in self.df.columns and pd.api.types.is_numeric_dtype(self.df.get('Cluster')):
                        f.write("\nDomain distribution by cluster (Top 3 domains per cluster):\n")
                        for i in sorted(cluster_counts_report.keys()): 
                            f.write(f"  Cluster {i} top domains:\n")
                            # Ensure cluster_df is from the self.df that has the 'Cluster' column
                            cluster_df_report = self.df[(self.df["Cluster"] == i) & (self.df["SenderEmail"].astype(str).str.strip() != "")]
                            
                            if cluster_df_report.empty:
                                f.write("    (No emails in this cluster with valid SenderEmail for domain analysis)\n")
                                continue

                            cluster_email_domains_report = []
                            for email_str in cluster_df_report["SenderEmail"]:
                                email_s_report = str(email_str).strip().lower()
                                if '@' in email_s_report:
                                    parts = email_s_report.split('@')
                                    if len(parts) == 2 and parts[1].strip() and '.' in parts[1].strip():
                                        cluster_email_domains_report.append(parts[1].strip())
                            
                            if not cluster_email_domains_report:
                                f.write("    (No valid domains found in this cluster's emails)\n")
                                continue

                            domain_counts_in_cluster_report = Counter(cluster_email_domains_report)
                            total_domains_in_cluster_count_report = len(cluster_email_domains_report)
                            
                            if not domain_counts_in_cluster_report:
                                f.write("    (No domains to report for this cluster)\n")
                            else:
                                for domain, count in domain_counts_in_cluster_report.most_common(3):
                                    percentage = (count / total_domains_in_cluster_count_report) * 100
                                    f.write(f"    {domain}: {count} emails ({percentage:.1f}%)\n")
                    else:
                         f.write("\n  Domain distribution by cluster: Skipped (Clustering incomplete, or 'SenderEmail'/'Cluster' column issue).\n")
                else:
                    f.write("  Clustering was not performed or yielded no results (e.g., insufficient data or TF-IDF errors).\n")

                f.write("\n\n=== VISUALIZING EMAIL CLUSTERS (PCA) ===\n")
                f.write("  A PCA plot ('email_clusters.png' or 'email_clusters_1D.png') is generated if data is sufficient.\n")
                f.write("  This visualizes subject-based email clusters. If not generated, check console logs from clustering step.\n")

                f.write("\n\n=== DOMAIN AGE ANALYSIS (For Top 25 Unique Domains) ===\n")
                if self.domain_creation_dates or any(age is not None for age in self.domain_ages.values()): 
                    f.write("Domain creation dates and ages (days) for successfully analyzed domains:\n")
                    # Sort by age, newest first for those with age
                    sorted_domain_ages_report = sorted(
                        [(domain, self.domain_creation_dates.get(domain), self.domain_ages.get(domain)) 
                         for domain in self.domain_ages if self.domain_ages.get(domain) is not None],
                        key=lambda x: x[2] # Sort by age (index 2)
                    )
                    for domain, cdate_obj, age in sorted_domain_ages_report:
                        cdate_str = cdate_obj.strftime('%Y-%m-%d') if cdate_obj else "N/A"
                        age_str = f"{age} days" if age is not None else "N/A" # age should not be None here
                        f.write(f"  {domain}: Created {cdate_str}, Age: {age_str}\n")
                    
                    unanalyzed_or_error_count = 0
                    for domain in self.domain_ages: # Iterate all domains attempted for WHOIS
                        if self.domain_ages[domain] is None: # Count those where age is None (error or no data)
                            unanalyzed_or_error_count +=1
                    if unanalyzed_or_error_count > 0:
                         f.write(f"  ...plus {unanalyzed_or_error_count} other domains were attempted for WHOIS but yielded no valid age data (check console for errors).\n")

                    if domain_age_stats_dict and domain_age_stats_dict.get("count",0) > 0 :
                        f.write(f"\nDomain age statistics (based on {domain_age_stats_dict['count']} domains with valid age data):\n")
                        avg_d = domain_age_stats_dict['avg_age_days']
                        min_d = domain_age_stats_dict['min_age_days']
                        max_d = domain_age_stats_dict['max_age_days']
                        f.write(f"  Average domain age: {avg_d:.1f} days (~{avg_d/365.25:.1f} years)\n")
                        f.write(f"  Newest domain: {min_d} days (~{min_d/365.25:.1f} years)\n")
                        f.write(f"  Oldest domain: {max_d} days (~{max_d/365.25:.1f} years)\n")
                    else:
                        f.write("\n  No valid domain age statistics calculated (e.g., all WHOIS lookups failed or returned no usable dates).\n")
                else:
                    f.write("  No domain age information could be retrieved or no domains were analyzed for age.\n")

                f.write("\n\n=== VIRUSTOTAL DOMAIN ANALYSIS (SAMPLE - Top 10 Unique, Valid Domains) ===\n")
                if virustotal_results: 
                    for domain, stats in virustotal_results.items():
                        if isinstance(stats, dict) and 'error' in stats: 
                            f.write(f"  {domain}: Error - {stats['error']}\n")
                        elif isinstance(stats, dict):
                             f.write(f"  {domain}: {stats}\n")
                        else: 
                            f.write(f"  {domain}: Invalid stats format received - {str(stats)}\n")
                else:
                    f.write("  No VirusTotal analysis performed or no results obtained (e.g., API key not provided, network issues, no valid domains to check, or all attempts failed).\n")

                f.write("\n\n=== HIGH-RISK EMAILS (FROM CSV 'Label' COLUMN) ===\n")
                if high_risk_emails_from_csv:
                    f.write(f"Found {len(high_risk_emails_from_csv)} high-risk emails based on 'Label' column's value being '1'.\n")
                    f.write("  Sample (up to 20):\n")
                    for email, reason in high_risk_emails_from_csv[:20]:
                        f.write(f"    {email}: {reason}\n")
                    if len(high_risk_emails_from_csv) > 20:
                        f.write(f"    ... and {len(high_risk_emails_from_csv) - 20} more.\n")
                else:
                    f.write("  No high-risk emails identified from CSV. This could be due to: 'Label' or 'SenderEmail' column missing, no rows with 'Label' as '1', or associated 'SenderEmail' was empty.\n")

                f.write("\n\n--- Optional Analyses (Placeholders) ---\n")
                f.write("  The following sections are placeholders for analyses not fully implemented/run in this script version:\n")
                f.write(f"  - Scam Categories in Email Content: {'Data available' if scam_categories_counter else 'Not run or no data'}\n")
                f.write(f"  - Top URL Domains in Email Bodies: {'Data available' if top_url_domains_list else 'Not run or no data'}\n")
                f.write(f"  - Comprehensive Threat Scores: {'Data available' if comprehensive_threat_scores else 'Not run or no data'}\n")


                f.write("\n\n--- End of Report ---")
            print(f"Report saved to '{report_filename}'")
        except IOError as e:
            print(f"Error: Could not write report file {report_filename}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during report generation: {e}")
            import traceback
            traceback.print_exc() 

# --- Helper function for plotting ---
def display_and_save_current_plot(title_text, filename):
    plt.title(title_text, fontsize=14) 
    try:
        plt.tight_layout(pad=1.5) 
    except Exception as e:
        # tight_layout can fail in some complex plot scenarios or with certain backends
        print(f"Note: plt.tight_layout() failed for '{filename}': {e}. Plot might have overlapping elements.")
    
    current_backend = plt.get_backend()
    is_interactive_backend = not any(b_name in current_backend.lower() for b_name in ['agg', 'ps', 'pdf', 'svg'])

    if is_interactive_backend:
        try:
            plt.show(block=False) 
            plt.pause(0.1) # Brief pause to allow GUI to render/update
        except Exception:
            # print(f"Note: Non-blocking plt.show() failed for '{filename}', using blocking. Close plot to continue.")
            plt.show() # Fallback to blocking show, user must close plot window.
    # If non-interactive (e.g. 'Agg'), plt.show() does nothing or errors, so it's skipped.

    try:
        if plt.get_fignums(): # Check if there are any active figures
            plt.savefig(filename, bbox_inches='tight') # bbox_inches='tight' helps fit everything
            print(f"Plot saved as {filename}")
        else:
            # This case might occur if plt.show() was blocking and user closed main script window first,
            # or if the plot creation itself failed.
            print(f"No active plot to save for {filename}. Window might have been closed, or plot creation failed.")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    finally:
        if plt.get_fignums(): 
            plt.close('all') # Close all figures to free memory and prevent state leakage

def main():
    file_path = input("Enter the path to your CSV file: ").strip()
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
        
    analyzer = EmailAnalyzer(file_path)
    analyzer.load_data()
    
    if analyzer.df is None: 
        print("Exiting due to data loading failure.")
        return

    analyzer.analyze_emails_and_usernames()

    print("\n=== 1. DOMAIN ANALYSIS ===")
    print("Counting domain frequencies...")
    valid_domains_for_top_list = [d for d in analyzer.domains if d and d.strip() and '.' in d]
    num_unique_domains = len(set(valid_domains_for_top_list))
    print(f"Found {num_unique_domains} unique domains (valid format from parsed emails)")
    
    top_10_domains = Counter(valid_domains_for_top_list).most_common(10)
    print("\nTop 10 sender domains:")
    if not top_10_domains:
        print("  No valid domain data to display.")
    else:
        for i, (domain, count) in enumerate(top_10_domains, 1):
            print(f"  {i}. {domain}: {count} emails")
        
        # Plotting
        if top_10_domains : # Ensure there's data to plot
            labels, counts = zip(*top_10_domains)
            plt.figure(figsize=(10, 8))
            plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, 
                    colors=sns.color_palette("Set2", len(labels)))
            display_and_save_current_plot("Top 10 Sender Domains", "s1_top_sender_domains.png")

    domain_entropy = analyzer.calculate_entropy(valid_domains_for_top_list) 
    max_possible_entropy = math.log2(num_unique_domains) if num_unique_domains > 0 else 0
    print(f"\nDomain distribution entropy: {domain_entropy:.2f}")
    if num_unique_domains > 0:
      print(f"  (Max possible entropy for these {num_unique_domains} unique domains: {max_possible_entropy:.2f})")
    print(f"  Interpretation: Higher entropy implies more diverse domains; lower implies concentration.")

    print("\n=== 2. EMAIL ADDRESS ANALYSIS ===")
    print("Counting email address frequencies...")
    valid_sender_emails = [e for e in analyzer.emails if e and e.strip() and '@' in e and e.split('@')[-1].strip() and '.' in e.split('@')[-1].strip()]
    num_unique_emails = len(set(valid_sender_emails))
    print(f"Found {num_unique_emails} unique email addresses (valid format from parsed emails)")

    top_10_emails = Counter(valid_sender_emails).most_common(10)
    print("\nTop 10 sender email addresses:")
    if not top_10_emails:
        print("  No valid email address data to display.")
    else:
        for i, (email, count) in enumerate(top_10_emails, 1):
            print(f"  {i}. {email}: {count} emails")
        
        if top_10_emails: # Ensure data for plot
            emails_plot, counts_plot = zip(*top_10_emails)
            plt.figure(figsize=(12, 8))
            sns.barplot(x=list(counts_plot), y=list(emails_plot), palette="viridis", orient='h')
            plt.xlabel("Number of Emails")
            plt.ylabel("Email Address")
            display_and_save_current_plot("Top 10 Sender Email Addresses", "s2_top_sender_emails.png")

    print("\nAnalyzing email address username patterns...")
    nums, underscores, avg_len, total_usernames_analyzed = analyzer.get_username_patterns()
    if total_usernames_analyzed > 0:
        print(f"  Usernames analyzed (from {total_usernames_analyzed} valid parsed emails):")
        print(f"    Containing numbers: {nums} ({nums/total_usernames_analyzed*100:.1f}%)")
        print(f"    Containing underscores: {underscores} ({underscores/total_usernames_analyzed*100:.1f}%)")
        print(f"    Average username length: {avg_len:.1f} characters")
        
        plot_data_usernames = {'Usernames w/ Numbers': nums, 'Usernames w/ Underscores': underscores}
        if any(v > 0 for v in plot_data_usernames.values()): 
            plt.figure(figsize=(8, 6))
            sns.barplot(x=list(plot_data_usernames.keys()), y=list(plot_data_usernames.values()), palette="pastel")
            plt.ylabel("Number of Usernames")
            display_and_save_current_plot("Username Pattern Counts", "s2_username_patterns_counts.png")
        else:
            print("    No usernames with numbers or underscores to plot.")
    else:
        print("  No usernames to analyze for patterns (e.g., 'SenderEmail' empty or all malformed).")

    print("\n=== 3. SUBJECT ACTION ANALYSIS ===")
    print("Analyzing subject lines for action-oriented language...")
    action_phrases_data, emails_with_actions, total_subjects_analyzed = analyzer.subject_action_analysis()
    
    if total_subjects_analyzed > 0:
        perc_emails_with_actions = (emails_with_actions / total_subjects_analyzed) * 100
        print(f"Found {emails_with_actions} emails ({perc_emails_with_actions:.1f}%) with action phrases in {total_subjects_analyzed} subjects.")
    else:
        print("No subjects to analyze for action phrases (e.g., 'Subject' column missing or empty).")

    print("\nTop action phrases in subject lines (up to 10):")
    if not action_phrases_data and total_subjects_analyzed > 0:
        print("  No pre-defined action phrases found.")
    elif not action_phrases_data: # Implies total_subjects_analyzed was 0 or issue in analysis
        print("  No subjects analyzed or no action phrases identified.")
    else: # action_phrases_data has content
        for i, (phrase, count, percentage) in enumerate(action_phrases_data, 1):
            print(f"  {i}. '{phrase}': {count} emails ({percentage:.1f}% of analyzed subjects)")
        
        if action_phrases_data: # Plot if data
            phrases_plot, counts_plot, _ = zip(*action_phrases_data)
            plt.figure(figsize=(10, 7))
            sns.barplot(x=list(counts_plot), y=list(phrases_plot), palette="coolwarm", orient='h')
            plt.xlabel("Number of Emails")
            plt.ylabel("Action Phrase")
            display_and_save_current_plot("Top Action Phrases in Subjects", "s3_top_action_phrases.png")
        
    print("\n=== 4. TEXT FEATURE EXTRACTION (TF-IDF from Subjects) ===")
    top_terms_from_tfidf = analyzer.extract_text_features() 
    
    print("\nTop terms across all email subjects (TF-IDF, up to 10):")
    if not top_terms_from_tfidf:
        print("  No TF-IDF terms to display (check console for TF-IDF process details).")
    else:
        for term, score in top_terms_from_tfidf:
            print(f"  {term}: (score) {score:.2f}")
        
        if top_terms_from_tfidf: # Plot if data
            terms_plot, scores_plot = zip(*top_terms_from_tfidf)
            plt.figure(figsize=(10, 7))
            sns.barplot(x=list(scores_plot), y=list(terms_plot), palette="magma", orient='h')
            plt.xlabel("TF-IDF Score")
            plt.ylabel("Term")
            display_and_save_current_plot("Top TF-IDF Terms from Subjects", "s4_top_tfidf_terms.png")

    print("\n=== 5. CLUSTERING EMAILS (K-Means on Subject TF-IDF) ===")
    cluster_words_info, pca_plot_data = analyzer.cluster_emails() 
    
    if analyzer.clusters is not None and len(analyzer.clusters) > 0:
        cluster_counts_for_plot = Counter(analyzer.clusters)
        if cluster_counts_for_plot: 
            labels = [f"Cluster {i}" for i in sorted(cluster_counts_for_plot.keys())] 
            sizes = [cluster_counts_for_plot[int(l.split()[-1])] for l in labels] 
            if sum(sizes) > 0: 
                plt.figure(figsize=(8, 8))
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                        colors=sns.color_palette("viridis", len(labels)))
                display_and_save_current_plot("Email Cluster Distribution (from Subjects TF-IDF)", "s5_cluster_distribution.png")
            else:
                print("  Cluster distribution plot skipped: No emails assigned to clusters.")
        else:
            print("  Cluster distribution plot skipped: No cluster count data available.")
    else:
        print("  Cluster distribution plot skipped: Clustering not performed or no results.")


    print("\n=== 6. VISUALIZING EMAIL CLUSTERS (PCA) ===")
    if pca_plot_data:
        reduced_features, clusters_for_pca = pca_plot_data
        if reduced_features.shape[1] == 2: 
            plt.figure(figsize=(10,6))
            scatter = plt.scatter(reduced_features[:,0], reduced_features[:,1], c=clusters_for_pca, cmap="viridis", alpha=0.7)
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            if len(np.unique(clusters_for_pca)) > 1: # Only add colorbar if multiple clusters
                 plt.colorbar(scatter, label='Cluster ID', ticks=np.unique(clusters_for_pca))
            else: # If only one cluster, colorbar is not very informative
                 plt.colorbar(scatter, label='Cluster ID (single cluster found)')
            display_and_save_current_plot("Email Clusters (PCA of Subjects TF-IDF)", "email_clusters.png")
        elif reduced_features.shape[1] == 1: 
            print("PCA produced only 1 component. Plotting 1D data distribution by cluster.")
            plt.figure(figsize=(10,max(4, len(np.unique(clusters_for_pca))))) # Adjust height for num clusters
            
            unique_pca_clusters = np.unique(clusters_for_pca)
            y_ticks_positions = []
            y_tick_labels = []

            for i, cluster_id in enumerate(unique_pca_clusters):
                cluster_data = reduced_features[clusters_for_pca == cluster_id, 0]
                # Add jitter for y-axis to separate points within the same cluster's "row"
                y_values = np.full_like(cluster_data, i) + np.random.normal(0, 0.05, size=cluster_data.shape[0])
                plt.scatter(cluster_data, y_values, label=f'Cluster {cluster_id}', alpha=0.7)
                y_ticks_positions.append(i)
                y_tick_labels.append(f"Cluster {cluster_id}")

            plt.xlabel("PCA Component 1")
            plt.yticks(y_ticks_positions, y_tick_labels)
            plt.ylabel("Cluster ID (with jitter for visualization)")
            if len(unique_pca_clusters) > 1 : plt.legend(title="Cluster ID")
            display_and_save_current_plot("Email Clusters (PCA - 1D of Subjects TF-IDF)", "email_clusters_1D.png")
        else:
             print("PCA plot not generated: PCA did not produce 1 or 2 components as expected.")
    else:
        print("PCA plot not generated (e.g. insufficient data, TF-IDF error, or PCA error). See logs from clustering step.")

    print("\n=== 7. DOMAIN AGE ANALYSIS (Top 25 Unique Domains) ===")
    analyzer.analyze_domain_age() 
    
    domain_stats = analyzer.get_domain_age_statistics()
    print("\nDomain age statistics (for domains where age could be retrieved):")
    if domain_stats and domain_stats.get('count', 0) > 0 :
        avg_days = domain_stats['avg_age_days']
        min_days = domain_stats['min_age_days']
        max_days = domain_stats['max_age_days']
        count_domains_aged = domain_stats['count']
        print(f"  (Based on {count_domains_aged} domains with valid age data)")
        print(f"  Average domain age: {avg_days:.1f} days (~{avg_days/365.25:.1f} years)")
        print(f"  Newest domain: {min_days:.0f} days (~{min_days/365.25:.1f} years)")
        print(f"  Oldest domain: {max_days:.0f} days (~{max_days/365.25:.1f} years)")
        
        stats_data_plot = {
            'Average Age (days)': avg_days,
            'Newest (days)': min_days,
            'Oldest (days)': max_days
        }
        if any(v > 0 for v in stats_data_plot.values()): # Plot if there's actual age data
            plt.figure(figsize=(8, 6))
            sns.barplot(x=list(stats_data_plot.keys()), y=list(stats_data_plot.values()), palette="crest")
            plt.ylabel("Age (days)")
            display_and_save_current_plot("Domain Age Summary Statistics", "s7_domain_age_summary_stats.png")
    else:
        print("  No domain age statistics to display (e.g., all WHOIS lookups failed or no domains analyzed).")

    valid_ages_for_hist = [age for age in analyzer.domain_ages.values() if age is not None and age >=0]
    if valid_ages_for_hist:
        plt.figure(figsize=(10, 6))
        num_bins = min(15, len(set(valid_ages_for_hist))) if len(set(valid_ages_for_hist)) > 0 else 1
        sns.histplot(valid_ages_for_hist, kde=True, bins=num_bins, color="skyblue")
        plt.xlabel("Domain Age (days)")
        plt.ylabel("Number of Domains")
        display_and_save_current_plot("Distribution of Analyzed Domain Ages", "s7_domain_age_distribution.png")
    elif domain_stats and domain_stats.get('count', 0) == 0: 
        print("  Domain age distribution plot skipped: No valid ages found for histogram.")


    # --- Optional analyses and report generation ---
    high_risk_emails_info = [] 
    scam_categories_info = Counter() # Default empty
    top_url_domains_info = []      # Default empty
    threat_scores_data = {}        # Default empty
    analyzer.virustotal_scores = {} # Ensure initialized/reset before potential new run

    run_optional_analysis = input("\nRun optional analyses (VirusTotal, High-Risk from CSV Label)? (yes/no): ").strip().lower()

    if run_optional_analysis == 'yes':
        api_key_vt = input("Enter your VirusTotal API key (leave blank to skip VT): ").strip()
        if api_key_vt:
            analyzer.analyze_with_virustotal(api_key_vt) 
            print("\n=== VIRUSTOTAL DOMAIN ANALYSIS (SAMPLE - Top 10 Unique, Valid Domains) ===")
            if analyzer.virustotal_scores:
                for domain, stats in analyzer.virustotal_scores.items():
                    if isinstance(stats, dict) and 'error' in stats:
                         print(f"  {domain}: Error - {stats['error']}")
                    elif isinstance(stats, dict):
                        print(f"  {domain}: {stats}")
                    else: 
                        print(f"  {domain}: Invalid/Unexpected VT stats format - {str(stats)}")
            else:
                print("  No VirusTotal analysis results (check logs from VT analysis step, API key, or network).")
        else:
            print("Skipping VirusTotal analysis as no API key was provided.")
            # analyzer.virustotal_scores remains {} as initialized

        print("\nIdentifying high-risk emails from CSV 'Label' column...")
        high_risk_emails_info = analyzer.identify_high_risk_emails_from_csv()
        print("\n=== HIGH-RISK EMAILS (FROM CSV 'Label' COLUMN) ===")
        if high_risk_emails_info:
            print(f"Found {len(high_risk_emails_info)} high-risk emails (Label=1). Displaying up to 20:")
            for email, reason in high_risk_emails_info[:20]: 
                print(f"  {email}: {reason}")
            if len(high_risk_emails_info) > 20:
                print(f"  ... and {len(high_risk_emails_info) - 20} more.")
        else:
            print("  No high-risk emails identified from CSV (check 'Label' column existence/values, 'SenderEmail', or console logs).")
    else: 
        print("Skipping optional analyses (VirusTotal, High-Risk from CSV).")
        # Ensure placeholders are empty if not run
        analyzer.virustotal_scores = {}
        high_risk_emails_info = []
    
    analyzer.generate_report_file(
        top_domains_list=top_10_domains,
        top_emails_list=top_10_emails,
        top_terms_list=top_terms_from_tfidf,
        cluster_words_output=cluster_words_info,
        domain_age_stats_dict=domain_stats,
        virustotal_results=analyzer.virustotal_scores, 
        high_risk_emails_from_csv=high_risk_emails_info,
        scam_categories_counter=scam_categories_info, 
        top_url_domains_list=top_url_domains_info,
        comprehensive_threat_scores=threat_scores_data
    )

if __name__ == "__main__":
    try:
        # Ensure matplotlib is imported early to catch backend issues if any
        import matplotlib
        print(f"Using Matplotlib backend: {matplotlib.get_backend()}")
        main()
    except Exception as e:
        print(f"A critical error occurred at the top level: {e}")
        import traceback
        traceback.print_exc()
        print("\nIf this is a Matplotlib backend issue (especially on headless systems or servers), you might need to configure it. For example, by setting `matplotlib.use('Agg')` at the beginning of the script (this will disable `plt.show()`) or by installing a GUI backend like 'python3-tk' (Linux) or 'pyqt5'.")