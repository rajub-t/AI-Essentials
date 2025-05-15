import pandas as pd
import numpy as np
import argparse
import datetime
import re
import string
import os
import sys
import unicodedata
from collections import Counter
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
import warnings

# Set plot backend to 'Agg' which doesn't require a display
plt.switch_backend('Agg')

# Suppress specific KMeans inertia warning when n_init is deprecated in future sklearn versions
warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto' in 1.4.", category=FutureWarning)


# --- Constants ---

# List of encodings to try when reading the file
ENCODINGS_TO_TRY = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

# Standardized column names expected by this script for analysis
REQUIRED_COLUMNS = ['sender_email', 'sender_domain', 'subject', 'body']

# Mapping for standardizing potential input column names to the REQUIRED_COLUMNS names
# Key: Potential input column name (case-insensitive and stripped)
# Value: Standard column name used internally by this script (from REQUIRED_COLUMNS)
COLUMN_MAPPING_ANALYZE = {
    'sender_email': 'sender_email',
    'senderemail': 'sender_email', # Handle variations like no underscore
    'sender e-mail': 'sender_email', # Handle spaces and hyphens

    'sender_domain': 'sender_domain',
    'senderdomain': 'sender_domain',
    'sender_dom': 'sender_domain', # Common abbreviation
    'SenderDomain': 'sender_domain', # User's specific capitalization from input file
    'sender domain': 'sender_domain', # Handle spaces

    'subject': 'subject',
    'subjectline': 'subject',
    'emailsubject': 'subject',

    'body': 'body',
    'emailbody': 'body',
    'content': 'body',
    'message': 'body',
}


# Value used to represent missing or invalid data
NULL_REPLACEMENT_VALUE = 'NULL'

# Keywords for detecting action-oriented language in subject lines
ACTION_KEYWORDS = [
    'urgent', 'action required', 'immediate action', 'important',
    'alert', 'warning', 'security alert', 'verify', 'update',
    'account suspended', 'invoice', 'payment', 'overdue', 'expire',
    'free', 'win', 'prize', 'claim', 'click here', 'download',
    'login', 'credentials', 'password', 'confirm', 'authorize'
    # Add more keywords relevant to malicious patterns
]

# TF-IDF Vectorization parameters
TFIDF_MAX_FEATURES = 1000
# Use sklearn's built-in English stop words
TFIDF_STOP_WORDS = 'english'

# K-Means Clustering parameters
KMEANS_N_CLUSTERS = 4
KMEANS_RANDOM_STATE = 42 # for reproducibility

# PCA Dimensionality Reduction parameters
PCA_N_COMPONENTS = 2
PCA_RANDOM_STATE = 42 # for reproducibility

# Output filenames
CLUSTER_PLOT_FILENAME = 'email_clusters.png'
REPORT_FILENAME = 'email_analysis_report.txt'

# Minimum number of samples required for clustering and PCA
# Need at least n_clusters samples for KMeans, and n_components + 1 for PCA
# Set to max requirement to ensure ML runs
MIN_SAMPLES_FOR_ML = max(KMEANS_N_CLUSTERS, PCA_N_COMPONENTS + 1)

# Minimum number of samples per cluster to attempt finding top domains for annotation
# LOWERED THIS THRESHOLD from 5 to 2 to include smaller clusters
MIN_CLUSTER_SIZE_FOR_ANNOTATION = 2


# --- Helper Functions ---

def calculate_domain_entropy(domain_list):
    """
    Calculates the Shannon entropy of a list of domains.
    Args:
        domain_list (list): A list of domain strings.
    Returns:
        float: The calculated entropy, or 0 if the list is empty or contains only one unique domain.
    """
    if not domain_list:
        return 0.0
    # Filter out NULLs and empty strings before counting
    valid_domains = [d for d in domain_list if isinstance(d, str) and d.strip() != '' and d != NULL_REPLACEMENT_VALUE and pd.notna(d)]
    if not valid_domains:
        return 0.0
    domain_counts = Counter(valid_domains)
    # Calculate probabilities
    probabilities = [count / len(valid_domains) for count in domain_counts.values()]
    # Calculate entropy (base 2 is common for information theory)
    return entropy(probabilities, base=2)

def clean_text(text):
    """
    Basic text cleaning: convert to string, handle NaN/None, lowercase, remove non-alphanumeric (except space).
    Args:
        text: Input text data.
    Returns:
        str: Cleaned text string.
    """
    if pd.isna(text) or text is None:
        return '' # Return empty string for missing values
    text = str(text).lower()
    # Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace (optional, but good practice)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_username_patterns(emails):
    """
    Analyzes patterns in the username part of email addresses.
    Looks for presence of numbers and underscores, and calculates average length.
    Args:
        emails (pd.Series): Series of email addresses.
    Returns:
        dict: Dictionary containing analysis results.
    """
    valid_usernames = []
    num_with_numbers = 0
    num_with_underscores = 0

    for email in emails:
        if isinstance(email, str) and '@' in email and email != NULL_REPLACEMENT_VALUE and email.strip() != '':
            try:
                # Ensure the part before @ is not empty
                username = email.split('@')[0]
                if username:
                    valid_usernames.append(username)
                    if re.search(r'\d', username):
                        num_with_numbers += 1
                    if '_' in username:
                        num_with_underscores += 1
            except Exception:
                 # Handle potential errors during split or regex for a single email
                 pass # Skip this email and continue

    total_valid_usernames = len(valid_usernames)
    # Calculate average length safely
    avg_length = np.mean([len(u) for u in valid_usernames]) if valid_usernames else 0

    return {
        'total_analyzed': len(emails), # Total rows passed into the function
        'valid_usernames_count': total_valid_usernames, # Count of emails with extractable usernames
        'num_with_numbers': num_with_numbers,
        'percent_with_numbers': (num_with_numbers / total_valid_usernames) * 100 if total_valid_usernames > 0 else 0.0,
        'num_with_underscores': num_with_underscores,
        'percent_with_underscores': (num_with_underscores / total_valid_usernames) * 100 if total_valid_usernames > 0 else 0.0,
        'average_length': avg_length
    }

def detect_action_words(subject_line):
    """
    Checks if a subject line contains any action-oriented keywords.
    Args:
        subject_line (str): The email subject string.
    Returns:
        bool: True if an action keyword is found, False otherwise.
    """
    if not isinstance(subject_line, str) or not subject_line.strip() or subject_line == NULL_REPLACEMENT_VALUE:
        return False
    subject_line = clean_text(subject_line) # Use cleaned version for checking

    # Check for exact word matches (more robust than substring)
    words = subject_line.split()
    if not words: # Handle empty string after cleaning
        return False

    for keyword in ACTION_KEYWORDS:
        keyword_words = keyword.lower().split()
        # Check if keyword is a single word match or a sequence of words
        for i in range(len(words) - len(keyword_words) + 1):
             if words[i:i+len(keyword_words)] == keyword_words:
                  return True
    return False

def generate_sample_data(filepath, num_rows=30):
    """
    Generates a sample CSV file for testing if the input file is not found.
    Args:
        filepath (str): The path where the sample file should be saved.
        num_rows (int): The number of sample rows to generate.
    """
    print(f"\nInput file not found. Generating sample data ({num_rows} rows) at {filepath}")

    # Use REQUIRED_COLUMNS for sample data structure
    sample_data = {col: [] for col in REQUIRED_COLUMNS}

    domains = ['example.com', 'mail.net', 'corp.org', 'phish.xyz', 'urgent-update.info', 'free-prize.co', 'bankofsim.com', 'support-client.net']
    subjects_legit = ['Meeting Tomorrow', 'Project Update', 'Weekly Report', 'Question about email', 'Your Order Confirmation', 'Regarding your request']
    subjects_action = ['Urgent: Action Required', 'Security Alert!', 'Your Account Needs Verification Immediately', 'Claim Your Prize Now!', 'Invoice Attached', 'Payment Overdue', 'Confirm Your Credentials', 'Download Your Report']
    bodies = [
        'Hi Team, Please review the attached document. Best regards.',
        'This is a routine update regarding your service. No action needed.',
        'We need your immediate attention on this matter. Click the link: http://malicious.link/login',
        'Congratulations! You have won. Download the file here: http://malicious.link/download',
        'Please find the invoice attached. Payment is overdue.',
        'Failure to respond within 24 hours will result in account suspension. Verify here: http://phishing.link/verify',
        'Click here to unsubscribe.', # Some benign action
        'Just following up on our conversation.' # Simple body
    ]

    # Ensure enough samples for ML even in sample data
    if num_rows < MIN_SAMPLES_FOR_ML:
         num_rows = MIN_SAMPLES_FOR_ML * 2 # Make sample data big enough

    for i in range(num_rows):
        # Mix legitimate-looking and suspicious patterns
        is_phishing_type = i % 3 == 0 or i % 5 == 1 # Roughly 1/3rd to 1/2 are 'phishing type'

        if is_phishing_type:
            domain = np.random.choice([d for d in domains if any(kw in d for kw in ['phish', 'urgent', 'free', 'bank', 'support'])])
            subject = np.random.choice(subjects_action)
            # Mix username patterns
            if i % 4 == 0: username = f"user{i//3 + 1}_support_{np.random.randint(100,999)}"
            elif i % 4 == 1: username = f"admin_{i//3 + 1}"
            else: username = f"support{np.random.randint(1, 10)}client"
            body = np.random.choice([b for b in bodies if any(kw in b for kw in ['click', 'urgent', 'account', 'invoice', 'verify', 'download'])])

        else: # Simulate more 'legit' emails
            domain = np.random.choice([d for d in domains if all(kw not in d for kw in ['phish', 'urgent', 'free', 'bank', 'support'])])
            subject = np.random.choice(subjects_legit)
            if i % 4 == 0: username = f"firstname.lastname{i}"
            elif i % 4 == 1: username = f"flast{i}"
            else: username = f"initial{np.random.randint(10, 99)}"
            body = np.random.choice([b for b in bodies if all(kw not in b for kw in ['click', 'urgent', 'account', 'invoice', 'verify', 'download'])])


        email = f"{username}@{domain}"

        sample_data['sender_email'].append(email)
        sample_data['sender_domain'].append(domain)
        sample_data['subject'].append(subject)
        sample_data['body'].append(body)

    sample_df = pd.DataFrame(sample_data)

    try:
        # Ensure directory exists
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sample_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Sample data saved to {filepath}")
    except IOError as e:
        print(f"Error saving sample data to {filepath}: {e}")
        # Exit if sample data cannot be saved, as loading will fail
        sys.exit(f"Failed to save sample data: {e}")


# --- EmailAnalyzer Class ---

class EmailAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.kmeans = None
        self.pca = None
        self.report_data = {}
        self.analysis_performed = False
        self.visualization_generated = False


    def load_data(self):
        """
        Loads data from the specified CSV file, trying multiple encodings.
        Handles FileNotFoundError by returning an error message.
        Applies column mapping and adds missing required columns with NULL values.
        Returns:
            bool: True if data was loaded (even if empty after processing), False otherwise.
        """
        print(f"Attempting to load file: {self.filepath}")

        # Check if file exists
        if not os.path.exists(self.filepath):
            print(f"Error: File not found: {self.filepath}")
            return False

        # Check file size first to avoid errors on empty files
        try:
            if os.path.exists(self.filepath) and os.path.getsize(self.filepath) == 0:
                print(f"Error: File is empty: {self.filepath}.")
                self.df = pd.DataFrame(columns=REQUIRED_COLUMNS)  # Create empty df with required columns
                return True  # Loading technically succeeded, but with no data
        except OSError as e:
            print(f"Error checking file size {self.filepath}: {e}. Attempting to load anyway.")

        for encoding in ENCODINGS_TO_TRY:
            try:
                # Assuming the processed output from the previous script is used,
                # it should be a standard CSV.
                self.df = pd.read_csv(self.filepath, encoding=encoding, low_memory=False)
                print(f"Successfully loaded file with encoding: {encoding}")

                # If data is loaded, apply column mapping before checking for required columns
                if not self.df.empty:
                    print("Applying column mapping...")
                    # Create a case-insensitive, space-agnostic map from actual columns to standard required names
                    current_cols_clean = {col.strip().lower(): col for col in self.df.columns}
                    rename_dict = {}
                    # Iterate through the standard required names we want
                    for standard_name in REQUIRED_COLUMNS:
                        # Find keys in COLUMN_MAPPING_ANALYZE whose value is the standard_name
                        potential_input_names_lower = [
                            k for k, v in COLUMN_MAPPING_ANALYZE.items() if v == standard_name
                        ]
                        # Check if any of the potential input names exist in the loaded columns (case/space agnostic)
                        found_match = False
                        for potential_lower in potential_input_names_lower:
                            if potential_lower in current_cols_clean:
                                original_col_name = current_cols_clean[potential_lower]
                                # Only add to rename if it's not already the standard name
                                if original_col_name != standard_name:
                                    rename_dict[original_col_name] = standard_name
                                found_match = True
                                break  # Found a match for this standard_name, move to the next one

                    if rename_dict:
                        print(f"Renaming columns: {rename_dict}")
                        self.df.rename(columns=rename_dict, inplace=True)
                    else:
                        print("No required columns needed renaming based on mapping.")

                # After mapping, check if all REQUIRED_COLUMNS are present using their standard names
                missing_cols = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
                if missing_cols:
                    print(f"Warning: Missing required columns after mapping: {', '.join(missing_cols)}. Adding with '{NULL_REPLACEMENT_VALUE}'.")
                    for col in missing_cols:
                        self.df[col] = NULL_REPLACEMENT_VALUE

                self.total_rows_loaded = len(self.df)  # Update total rows after loading
                return True  # Loading succeeded (even if resulting df is empty)

            except UnicodeDecodeError:
                print(f"Failed to decode with encoding: {encoding}")
            except pd.errors.EmptyDataError:
                print(f"File is readable but contains no data with encoding: {encoding}. Proceeding with empty data.")
                self.df = pd.DataFrame(columns=REQUIRED_COLUMNS)  # Create empty df with required columns
                return True  # Loading technically succeeded, but with empty data
            except Exception as e:
                print(f"An error occurred while loading with encoding {encoding}: {e}")

        # If loop finishes without returning True, it means loading failed with all encodings
        print(f"Error: Could not load file '{self.filepath}' with any of the attempted encodings or default settings.")
        # Ensure self.df is an empty dataframe with required columns on complete failure
        if self.df is None or self.df.empty:
            self.df = pd.DataFrame(columns=REQUIRED_COLUMNS)
            print("Initializing empty dataframe with required columns.")

        return False  # Explicitly indicate loading failed to get meaningful data


    def _preprocess_text(self):
        """
        Combines subject and body text, cleans it, and prepares for vectorization.
        Updates the internal dataframe by filtering out rows with empty text after cleaning.
        Assumes 'subject' and 'body' columns exist (added by load_data if missing).
        Returns:
            pd.Series: Series of combined and cleaned text corresponding to the filtered dataframe,
                       or None if required text columns are missing or results in insufficient data for ML.
        """
        if self.df is None or self.df.empty:
            print("No data to preprocess text.")
            return None

        # Ensure subject and body columns exist (added by load_data if missing)
        subject_col = 'subject'
        body_col = 'body'

        # Check if required text columns exist after loading/adding
        if subject_col not in self.df.columns or body_col not in self.df.columns:
            print(f"Missing required text columns '{subject_col}' or '{body_col}'. Cannot perform text analysis.")
            # Ensure df still has other required columns if text ones were missing
            for col in REQUIRED_COLUMNS:
                 if col not in self.df.columns:
                      self.df[col] = NULL_REPLACEMENT_VALUE
            return None


        print("Preprocessing text (combining subject and body)...")
        # Fill potential NaNs in original columns with empty strings before combining and cleaning
        # Ensure columns are string type before string operations
        self.df[subject_col] = self.df[subject_col].fillna('').astype(str)
        self.df[body_col] = self.df[body_col].fillna('').astype(str)

        # Combine subject and body with a separator
        combined_text = self.df[subject_col] + " " + self.df[body_col]

        # Apply cleaning to the combined text
        cleaned_text = combined_text.apply(clean_text)

        # Filter out rows where cleaning resulted in empty text (will cause issues with TF-IDF)
        non_empty_text_mask = cleaned_text.str.strip().astype(bool)
        rows_removed = len(cleaned_text) - non_empty_text_mask.sum()
        if rows_removed > 0:
            print(f"Filtered out {rows_removed} rows with empty text after cleaning.")
            self.df = self.df[non_empty_text_mask].copy() # Filter the main dataframe in place
            cleaned_text = cleaned_text[non_empty_text_mask].copy() # Filter the text series and ensure copy


        if self.df.empty:
             print("No data remaining after filtering empty text. Cannot perform ML analysis.")
             return None

        # Check minimum samples for ML *after* filtering
        if len(self.df) < MIN_SAMPLES_FOR_ML:
             print(f"Insufficient data for ML analysis ({len(self.df)} rows). Need at least {MIN_SAMPLES_FOR_ML} rows.")
             return None

        # Reset index after filtering to avoid issues with subsequent merges/alignments
        self.df = self.df.reset_index(drop=True)
        cleaned_text = cleaned_text.reset_index(drop=True)


        return cleaned_text

    def _vectorize_text(self, cleaned_text):
        """
        Applies TF-IDF vectorization to cleaned text data.
        Args:
            cleaned_text (pd.Series): Series of cleaned text strings.
        Returns:
            sparse matrix: TF-IDF matrix, or None if vectorization fails or input is insufficient.
        """
        if cleaned_text is None or cleaned_text.empty:
            print("No cleaned text data for vectorization.")
            self.tfidf_matrix = None
            return None

        print("Applying TF-IDF vectorization...")
        try:
            # Ensure enough samples for TF-IDF fit (though TfidfVectorizer handles min_df, min samples is implicit)
            # Check against MIN_SAMPLES_FOR_ML again, as _preprocess_text might have returned data below threshold
            if len(cleaned_text) < MIN_SAMPLES_FOR_ML:
                 print(f"Insufficient samples ({len(cleaned_text)}) for TF-IDF vectorization.")
                 self.tfidf_matrix = None
                 return None

            self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words=TFIDF_STOP_WORDS)
            self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_text)
            print(f"TF-IDF matrix created with shape: {self.tfidf_matrix.shape}")
            return self.tfidf_matrix
        except Exception as e:
            print(f"Error during TF-IDF vectorization: {e}")
            self.tfidf_matrix = None
            return None


    def _cluster_data(self):
        """
        Applies K-means clustering to the TF-IDF matrix.
        Adds 'cluster' labels to the dataframe.
        Returns:
            bool: True if clustering was successful, False otherwise.
        """
        if self.tfidf_matrix is None:
            print("No TF-IDF matrix available for clustering.")
            # Add cluster column with NULL if clustering couldn't run
            self.df['cluster'] = NULL_REPLACEMENT_VALUE
            self.kmeans = None
            return False

        print(f"Applying K-means clustering with {KMEANS_N_CLUSTERS} clusters...")
        try:
            # Ensure enough samples for clustering (already checked before vectorization, but belt-and-suspenders)
            if self.tfidf_matrix.shape[0] < KMEANS_N_CLUSTERS:
                 print(f"Not enough samples ({self.tfidf_matrix.shape[0]}) for {KMEANS_N_CLUSTERS} clusters.")
                 # Assign a single cluster label if possible (e.g., all to 0 if samples > 0), or NULL
                 if self.tfidf_matrix.shape[0] > 0:
                     self.df['cluster'] = 0
                     print("Assigned all samples to cluster 0.")
                 else: # Should be caught by empty df check earlier, but safety
                     self.df['cluster'] = NULL_REPLACEMENT_VALUE
                     print("No samples to assign clusters.")
                 self.kmeans = None # Clustering did not truly run
                 return False # Clustering technically failed due to insufficient data

            # Using n_init='auto' for newer sklearn versions to avoid FutureWarnings
            # Fallback to n_init=10 for compatibility if needed, but 'auto' is preferred
            self.kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init='auto')
            cluster_labels = self.kmeans.fit_predict(self.tfidf_matrix)
            self.df['cluster'] = cluster_labels
            print("K-means clustering complete.")
            return True
        except Exception as e:
            print(f"Error during K-means clustering: {e}")
            self.df['cluster'] = NULL_REPLACEMENT_VALUE # Assign NULL if clustering failed
            self.kmeans = None
            return False

    def _reduce_dimensions(self):
        """
        Applies PCA to the TF-IDF matrix for dimensionality reduction to 2 components.
        Adds 'pc1' and 'pc2' columns to the dataframe.
        Returns:
            bool: True if PCA was successful, False otherwise.
        """
        if self.tfidf_matrix is None:
            print("No TF-IDF matrix available for PCA.")
            self.df['pc1'] = NULL_REPLACEMENT_VALUE
            self.df['pc2'] = NULL_REPLACEMENT_VALUE
            self.pca = None
            return False

        print(f"Applying PCA for dimensionality reduction to {PCA_N_COMPONENTS} components...")
        try:
            # Ensure enough samples for PCA (needs n_components + 1)
            if self.tfidf_matrix.shape[0] < PCA_N_COMPONENTS + 1:
                 print(f"Not enough samples ({self.tfidf_matrix.shape[0]}) for {PCA_N_COMPONENTS} PCA components.")
                 self.df['pc1'] = NULL_REPLACEMENT_VALUE
                 self.df['pc2'] = NULL_REPLACEMENT_VALUE
                 self.pca = None # PCA did not truly run
                 return False # PCA technically failed due to insufficient data


            self.pca = PCA(n_components=PCA_N_COMPONENTS, random_state=PCA_RANDOM_STATE)
            # PCA fit_transform works on dense data, convert sparse matrix to array
            principal_components = self.pca.fit_transform(self.tfidf_matrix.toarray())

            self.df['pc1'] = principal_components[:, 0]
            self.df['pc2'] = principal_components[:, 1]
            print("PCA dimensionality reduction complete.")
            return True
        except Exception as e:
            print(f"Error during PCA dimensionality reduction: {e}")
            self.df['pc1'] = NULL_REPLACEMENT_VALUE
            self.df['pc2'] = NULL_REPLACEMENT_VALUE
            self.pca = None
            return False

    def analyze_domains(self, top_n=10):
        """
        Analyzes sender domains to find top sources and calculate overall entropy.
        Args:
            top_n (int): Number of top domains to report.
        """
        # Check if required column exists after loading/filtering/mapping
        if self.df is None or self.df.empty or 'sender_domain' not in self.df.columns:
            self.report_data['domain_analysis'] = "--- Domain Analysis ---\n  Domain analysis skipped: No data or 'sender_domain' column missing."
            return

        print("Analyzing sender domains...")
        # Use the dataframe *after* filtering by _preprocess_text if that happened
        # Ensure sender_domain is string type before operations
        self.df['sender_domain'] = self.df['sender_domain'].astype(str).replace('nan', '', regex=False)
        valid_domains = self.df['sender_domain'].dropna().tolist()
        valid_domains = [d for d in valid_domains if d.strip() != '' and d != NULL_REPLACEMENT_VALUE]

        if not valid_domains:
            self.report_data['domain_analysis'] = "--- Domain Analysis ---\n  Domain analysis: No valid domains found."
            return

        # Top Domains
        domain_counts = pd.Series(valid_domains).value_counts()
        top_domains = domain_counts.head(top_n)

        # Overall Entropy
        overall_entropy = calculate_domain_entropy(valid_domains)

        report_lines = [
            "--- Domain Analysis ---",
            f"Total domains analyzed (after filtering): {len(valid_domains)}",
            f"Unique domains: {len(domain_counts)}",
            f"Overall Domain Entropy (base 2): {overall_entropy:.4f}",
            "\nTop Sender Domains:",
        ]
        if top_domains.empty:
             report_lines.append("  No top domains found.")
        else:
            for domain, count in top_domains.items():
                report_lines.append(f"  {domain}: {count}")

        self.report_data['domain_analysis'] = "\n".join(report_lines)
        print("Domain analysis complete.")


    def analyze_senders(self, top_n=10):
        """
        Analyzes sender email addresses to find the most common senders.
        Args:
            top_n (int): Number of most common senders to report.
        """
        # Check if required column exists after loading/filtering/mapping
        if self.df is None or self.df.empty or 'sender_email' not in self.df.columns:
            self.report_data['sender_analysis'] = "--- Sender Analysis ---\n  Sender analysis skipped: No data or 'sender_email' column missing."
            return

        print("Analyzing sender email addresses...")
        # Use the dataframe *after* filtering by _preprocess_text if that happened
        # Ensure sender_email is string type before operations
        self.df['sender_email'] = self.df['sender_email'].astype(str).replace('nan', '', regex=False)
        valid_emails = self.df['sender_email'].dropna().tolist()
        valid_emails = [e for e in valid_emails if e.strip() != '' and e != NULL_REPLACEMENT_VALUE and '@' in e]

        if not valid_emails:
            self.report_data['sender_analysis'] = "--- Sender Analysis ---\n  Sender analysis: No valid email addresses found."
            return

        # Most Common Senders
        email_counts = pd.Series(valid_emails).value_counts()
        most_common_senders = email_counts.head(top_n)

        report_lines = [
            "--- Sender Analysis ---",
            f"Total email addresses analyzed (after filtering): {len(valid_emails)}",
            f"Unique email addresses: {len(email_counts)}",
            "\nMost Common Senders:",
        ]
        if most_common_senders.empty:
             report_lines.append("  No common senders found.")
        else:
            for email, count in most_common_senders.items():
                report_lines.append(f"  {email}: {count}")

        self.report_data['sender_analysis'] = "\n".join(report_lines)
        print("Sender analysis complete.")

    def analyze_username_patterns(self):
        """
        Analyzes patterns in the username part of sender email addresses.
        Reports on presence of numbers, underscores, and average length.
        """
        # Check if required column exists after loading/filtering/mapping
        if self.df is None or self.df.empty or 'sender_email' not in self.df.columns:
            self.report_data['username_pattern_analysis'] = "--- Username Pattern Analysis ---\n  Username pattern analysis skipped: No data or 'sender_email' column missing."
            return

        print("Analyzing username patterns...")
        # Use the dataframe *after* filtering by _preprocess_text if that happened
        # Ensure sender_email is string type before operations
        self.df['sender_email'] = self.df['sender_email'].astype(str).replace('nan', '', regex=False)
        valid_emails = self.df['sender_email'].dropna()
        valid_emails = valid_emails[(valid_emails != NULL_REPLACEMENT_VALUE) & (valid_emails.str.strip() != '') & (valid_emails.str.contains('@'))]

        if valid_emails.empty:
             self.report_data['username_pattern_analysis'] = "--- Username Pattern Analysis ---\n  Username pattern analysis: No valid email addresses found with usernames."
             return

        pattern_results = analyze_username_patterns(valid_emails)

        report_lines = [
            "--- Username Pattern Analysis ---",
            f"Email addresses with valid username part analyzed: {pattern_results['valid_usernames_count']}",
            f"Number of usernames containing numbers: {pattern_results['num_with_numbers']} ({pattern_results['percent_with_numbers']:.2f}%)",
            f"Number of usernames containing underscores: {pattern_results['num_with_underscores']} ({pattern_results['percent_with_underscores']:.2f}%)",
            f"Average username length: {pattern_results['average_length']:.2f}"
        ]

        self.report_data['username_pattern_analysis'] = "\n".join(report_lines)
        print("Username pattern analysis complete.")


    def detect_action_language(self):
        """
        Detects action-oriented language in subject lines.
        Reports the count and percentage of subjects containing keywords.
        """
        # Check if required column exists after loading/filtering/mapping
        if self.df is None or self.df.empty or 'subject' not in self.df.columns:
            self.report_data['action_language_analysis'] = "--- Action Language Analysis (Subjects) ---\n  Action language analysis skipped: No data or 'subject' column missing."
            return

        print("Detecting action-oriented language in subjects...")
        # Use the dataframe *after* filtering by _preprocess_text if that happened
        # Ensure subject is string type before operations
        self.df['subject'] = self.df['subject'].astype(str).replace('nan', '', regex=False)
        subjects = self.df['subject'].dropna()
        subjects = subjects[(subjects != NULL_REPLACEMENT_VALUE) & (subjects.str.strip() != '')]

        if subjects.empty:
             self.report_data['action_language_analysis'] = "--- Action Language Analysis (Subjects) ---\n  Action language analysis: No valid subject lines found."
             return

        total_subjects = len(subjects)
        subjects_with_action_words = subjects.apply(detect_action_words).sum()
        percent_with_action_words = (subjects_with_action_words / total_subjects) * 100 if total_subjects > 0 else 0.0

        report_lines = [
            "--- Action Language Analysis (Subjects) ---",
            f"Total subjects analyzed (after filtering): {total_subjects}",
            f"Subjects containing action keywords: {subjects_with_action_words} ({percent_with_action_words:.2f}%)",
            f"Sample Keywords checked: {', '.join(ACTION_KEYWORDS[:5])}{'...' if len(ACTION_KEYWORDS) > 5 else ''}" # Show a few keywords
        ]

        self.report_data['action_language_analysis'] = "\n".join(report_lines)
        print("Action language analysis complete.")


    def perform_analysis(self):
        """
        Orchestrates all analysis steps including text processing, ML, and pattern analysis.
        Returns:
            bool: True if ML analysis steps required for visualization completed successfully
                  and there is data to visualize, False otherwise.
        """
        if self.df is None or self.df.empty:
            print("No data loaded. Cannot perform analysis.")
            self.analysis_performed = False
            return False # Cannot perform any analysis

        print("\n--- Starting Email Analysis ---")
        # Store initial row count for the report
        initial_row_count = len(self.df)
        self.report_data['total_rows_initially_loaded'] = initial_row_count


        # 1. Text Preprocessing (filters the dataframe)
        # Returns cleaned text *and* updates self.df in place by filtering
        cleaned_text = self._preprocess_text()

        # Store row count after preprocessing/filtering
        rows_after_preprocessing = len(self.df)
        self.report_data['rows_after_preprocessing'] = rows_after_preprocessing


        # Check if sufficient data remains for ML after preprocessing
        if cleaned_text is None or rows_after_preprocessing < MIN_SAMPLES_FOR_ML:
             print("Insufficient data remaining after preprocessing for ML analysis. Skipping ML steps.")
             # Still proceed with pattern analysis on the reduced df if it's not entirely empty
             if not self.df.empty:
                  self.analyze_domains()
                  self.analyze_senders()
                  self.analyze_username_patterns()
                  self.detect_action_language()
                  self.analysis_performed = True # Pattern analysis was performed
                  return False # Indicate ML for viz failed
             else:
                  self.analysis_performed = False
                  return False # No data left at all, ML for viz failed


        # 2. Text Vectorization (TF-IDF)
        tfidf_success = self._vectorize_text(cleaned_text)
        if tfidf_success is None:
             print("TF-IDF vectorization failed. Cannot proceed with Clustering/PCA.")
             # Still proceed with pattern analysis on the reduced df
             self.analyze_domains()
             self.analyze_senders()
             self.analyze_username_patterns()
             self.detect_action_language()
             self.analysis_performed = True # Pattern analysis was performed
             return False # Indicate ML for viz failed


        # 3. Clustering (K-means)
        clustering_successful = self._cluster_data() # Adds 'cluster' column

        # 4. Dimensionality Reduction (PCA for visualization)
        pca_successful = self._reduce_dimensions() # Adds 'pc1', 'pc2' columns


        # 5. Pattern Analysis (runs on the dataframe *after* preprocessing/filtering)
        # These analyses use the data remaining after text filtering
        self.analyze_domains()
        self.analyze_senders()
        self.analyze_username_patterns()
        self.detect_action_language()


        print("--- Email Analysis Complete ---")
        self.analysis_performed = True # ML was attempted

        # Check if ML steps required for visualization completed successfully AND there is data to plot
        ml_for_viz_successful = clustering_successful and pca_successful and 'cluster' in self.df.columns and 'pc1' in self.df.columns and 'pc2' in self.df.columns and not self.df.empty

        return ml_for_viz_successful # Return status indicating if data is ready for visualization


    def generate_visualization(self):
        """
        Generates a 2D PCA plot showing email clusters.
        Annotates clusters with top sender domains.
        Saves the plot to a file.
        Returns:
             bool: True if visualization was successfully generated and saved, False otherwise.
        """
        # Check if necessary data and columns for visualization exist
        required_viz_cols = ['pc1', 'pc2', 'cluster', 'sender_domain']
        if self.df is None or self.df.empty or not all(col in self.df.columns for col in required_viz_cols):
             print(f"Cannot generate visualization: Missing data or required columns after analysis: {required_viz_cols}.")
             self.visualization_generated = False
             return False

        # Ensure there's more than one unique cluster label and sufficient data points for scatter plot
        # Filter out potential NULL clusters before counting unique
        valid_clusters_in_df = self.df['cluster'].dropna().unique()
        if len(valid_clusters_in_df) < 2 or len(self.df) < PCA_N_COMPONENTS + 1: # Need at least PCA_N_COMPONENTS + 1 samples for PCA
             print(f"Cannot generate visualization: Need at least 2 unique clusters and {PCA_N_COMPONENTS + 1} data points.")
             self.visualization_generated = False
             return False


        print(f"\nGenerating visualization: {CLUSTER_PLOT_FILENAME}...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each cluster separately to handle colors and legend
        # Filter out rows where cluster is NULL_REPLACEMENT_VALUE before plotting
        df_to_plot = self.df[self.df['cluster'] != NULL_REPLACEMENT_VALUE].copy()

        # Ensure cluster column is numeric type for cmap indexing if possible, or handle mapping string labels to colors
        # Assuming K-Means produced integer labels, but handle if they became strings '0', '1', etc.
        df_to_plot['cluster_numeric'] = pd.to_numeric(df_to_plot['cluster'], errors='coerce').fillna(-1).astype(int)
        # Filter out rows where conversion failed (-1)
        df_to_plot = df_to_plot[df_to_plot['cluster_numeric'] != -1].copy()

        if df_to_plot.empty:
             print("No plottable data after filtering NULL/invalid clusters.")
             self.visualization_generated = False
             plt.close(fig)
             return False

        unique_plottable_clusters = sorted(df_to_plot['cluster_numeric'].unique())
        if len(unique_plottable_clusters) < 2:
             print("Not enough unique clusters remaining after filtering for plotting.")
             self.visualization_generated = False
             plt.close(fig)
             return False

        # Use a colormap suitable for distinct clusters based on the actual number of unique clusters plotted
        cmap = plt.get_cmap('viridis', len(unique_plottable_clusters))
        colors = [cmap(i) for i in range(len(unique_plottable_clusters))] # Get colors based on index in sorted unique list

        # Map numeric cluster labels to colors for plotting
        cluster_color_map = {label: colors[i] for i, label in enumerate(unique_plottable_clusters)}
        plot_colors = df_to_plot['cluster_numeric'].map(cluster_color_map)


        scatter = ax.scatter(df_to_plot['pc1'], df_to_plot['pc2'],
                            c=plot_colors, # Use mapped colors
                            s=30, # Slightly larger points
                            alpha=0.7, # Slightly less transparent
                            edgecolors='w', linewidth=0.5) # Add white edge for better visibility


        # Add annotations for top domains per cluster (using original cluster labels)
        for cluster_label in sorted(valid_clusters_in_df): # Iterate through valid original cluster labels
            cluster_df = self.df[self.df['cluster'] == cluster_label].copy() # Use original df for filtering


            # Check cluster size against the (lowered) annotation threshold
            if len(cluster_df) < MIN_CLUSTER_SIZE_FOR_ANNOTATION:
                 # print(f"Skipping annotation for cluster {cluster_label}: Insufficient samples ({len(cluster_df)})")
                 continue # Skip annotation if cluster is too small

            # Check if centroid calculation is possible for this cluster's PCA points
            if cluster_df[['pc1', 'pc2']].dropna().empty:
                 print(f"Skipping annotation placement for cluster {cluster_label}: Centroid calculation failed (likely missing PCA values).")
                 continue


            # Get top 3 domains for this cluster
            # Ensure 'sender_domain' column exists and handle NULLs/empty strings
            cluster_domains = cluster_df['sender_domain'].dropna()
            cluster_domains = cluster_domains[(cluster_domains != NULL_REPLACEMENT_VALUE) & (cluster_domains.str.strip() != '')]

            annotation_text = f"Cluster {cluster_label}"
            if not cluster_domains.empty:
                top_domains = cluster_domains.value_counts().head(3).index.tolist()
                annotation_text += "\nTop Domains:\n" + "\n".join(top_domains)
            else:
                 annotation_text += "\nNo valid domains"


            # Calculate centroid for annotation placement using PCA components from the original cluster rows
            centroid_pc1 = cluster_df['pc1'].mean()
            centroid_pc2 = cluster_df['pc2'].mean()


            # Annotate near the centroid
            try:
                 ax.annotate(annotation_text,
                             (centroid_pc1, centroid_pc2),
                             textcoords="offset points",
                             xytext=(15, 15), # Offset text slightly more
                             ha='left',
                             fontsize=9, # Slightly smaller font for annotations
                             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8, edgecolor='lightgray')) # Add a light border

            except Exception as e:
                 print(f"Error annotating cluster {cluster_label}: {e}")


        # Labels, Title, and Legend
        ax.set_xlabel("Principal Component 1", fontsize=12)
        ax.set_ylabel("Principal Component 2", fontsize=12)
        ax.set_title("Email Clusters (PCA)", fontsize=14)

        # Add legend
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
                                    markerfacecolor=cluster_color_map[label], markersize=10)
                         for label in unique_plottable_clusters]
        ax.legend(handles=legend_handles, title="Clusters")

        ax.grid(True, linestyle='--', alpha=0.6) # Add a subtle grid

        # --- Set X-axis limit ---
        # Set the upper limit of the x-axis to 0.6, keep lower limit automatic
        ax.set_xlim(right=0.6)
        print("Set x-axis upper limit to 0.6.")
        # --- End Set X-axis limit ---


        # Save the plot
        try:
            # Ensure output directory exists (same as input file)
            output_dir = os.path.dirname(self.filepath)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)
            if not output_dir: # If input file is in the current directory
                 output_dir = '.'

            save_path = os.path.join(output_dir, CLUSTER_PLOT_FILENAME)
            plt.savefig(save_path, bbox_inches='tight', dpi=300) # Use higher DPI for better resolution
            print(f"Visualization saved to {save_path}")
            self.visualization_generated = True
        except Exception as e:
            print(f"Error saving visualization to {save_path}: {e}")
            self.visualization_generated = False
        finally:
            plt.close(fig) # Close the figure to free memory

        return self.visualization_generated


    def generate_report(self):
        """Generates a comprehensive text report with detailed analysis findings."""
        print("\n--- Generating Detailed Analysis Report ---")
        report_lines = []

        # Header Section
        report_lines.append("--- Detailed Email Analysis Report ---")
        report_lines.append(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Input File: {self.filepath}")
        report_lines.append("-" * 60)

        # General Statistics
        report_lines.append("General Statistics:")
        report_lines.append(f"  Total rows initially loaded: {self.report_data.get('total_rows_initially_loaded', 'N/A')}")
        report_lines.append(f"  Rows remaining after text preprocessing/filtering: {self.report_data.get('rows_after_preprocessing', 'N/A')}")
        report_lines.append("-" * 60)

        # Domain Analysis
        if 'domain_analysis' in self.report_data:
            report_lines.append(self.report_data['domain_analysis'])
        else:
            report_lines.append("--- Domain Analysis ---\n  Analysis not performed or data missing.")
        report_lines.append("-" * 60)

        # Sender Analysis
        if 'sender_analysis' in self.report_data:
            report_lines.append(self.report_data['sender_analysis'])
        else:
            report_lines.append("--- Sender Analysis ---\n  Analysis not performed or data missing.")
        report_lines.append("-" * 60)

        # Username Pattern Analysis
        if 'username_pattern_analysis' in self.report_data:
            report_lines.append(self.report_data['username_pattern_analysis'])
        else:
            report_lines.append("--- Username Pattern Analysis ---\n  Analysis not performed or data missing.")
        report_lines.append("-" * 60)

        # Action Language Analysis
        if 'action_language_analysis' in self.report_data:
            report_lines.append(self.report_data['action_language_analysis'])
        else:
            report_lines.append("--- Action Language Analysis (Subjects) ---\n  Analysis not performed or data missing.")
        report_lines.append("-" * 60)

        # ML Analysis Summary
        report_lines.append("--- Machine Learning Analysis Summary ---")
        if self.analysis_performed and self.tfidf_matrix is not None and self.kmeans is not None and self.pca is not None:
            report_lines.append(f"  Text vectorization performed (TF-IDF):")
            report_lines.append(f"    - Samples: {self.tfidf_matrix.shape[0]}")
            report_lines.append(f"    - Features: {self.tfidf_matrix.shape[1]}")
            report_lines.append(f"  Clustering performed (K-Means):")
            report_lines.append(f"    - Number of clusters: {KMEANS_N_CLUSTERS}")
            report_lines.append(f"  Dimensionality reduction performed (PCA):")
            report_lines.append(f"    - Components: {PCA_N_COMPONENTS}")

            # Cluster Sizes and Top Domains
            if 'cluster' in self.df.columns and not self.df.empty:
                valid_clusters_in_df = self.df['cluster'].dropna()
                if not valid_clusters_in_df.empty:
                    cluster_sizes = valid_clusters_in_df.value_counts().sort_index()
                    report_lines.append("\n  Cluster Sizes and Top Domains (after filtering):")
                    for cluster_id, size in cluster_sizes.items():
                        report_lines.append(f"    - Cluster {cluster_id}: {size} emails")
                        # Get top domains for this cluster
                        cluster_df = self.df[self.df['cluster'] == cluster_id]
                        cluster_domains = cluster_df['sender_domain'].dropna()
                        cluster_domains = cluster_domains[(cluster_domains != NULL_REPLACEMENT_VALUE) & (cluster_domains.str.strip() != '')]
                        top_domains = cluster_domains.value_counts().head(3)
                        if not top_domains.empty:
                            report_lines.append("      Top Domains:")
                            for domain, count in top_domains.items():
                                report_lines.append(f"        - {domain}: {count}")
                        else:
                            report_lines.append("      No valid domains found.")
                else:
                    report_lines.append("    - No valid clusters found in remaining data.")
            else:
                report_lines.append("    - Cluster information not available.")

            if self.visualization_generated:
                report_lines.append(f"  Visualization saved to: {CLUSTER_PLOT_FILENAME}")
            else:
                report_lines.append("  Visualization not generated due to insufficient data or errors in ML steps.")
        else:
            report_lines.append("  ML Analysis (TF-IDF, Clustering, PCA) not performed due to insufficient data or errors.")
        report_lines.append("-" * 60)

        # End of Report
        report_lines.append("End of Detailed Report")

        # Store the report text
        self.report_data['report_text'] = "\n".join(report_lines)
        print("Detailed report generation complete (content stored internally).")

        # Display the full report in the console
        print("\n--- Full Detailed Report ---")
        print(self.report_data['report_text'])


    def save_report(self, filename):
        """Saves the generated report to a text file."""
        if 'report_text' not in self.report_data:
            print("No report data generated yet.")
            return

        try:
            # Ensure directory exists
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir)

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.report_data['report_text'])
            print(f"Analysis report saved successfully to: {filename}")
        except IOError as e:
            print(f"Error saving report to {filename}: {e}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Analyze email data for malicious patterns and visualize clusters.")
    parser.add_argument('filepath', help="Path to the input CSV email file.")
    args = parser.parse_args()

    analyzer = EmailAnalyzer(args.filepath)

    # 1. Load data (generates sample if file doesn't exist, handles empty files)
    loading_successful = analyzer.load_data()

    if not loading_successful or analyzer.df is None or analyzer.df.empty:
        print("Data loading failed or resulted in an empty dataset. Cannot proceed with analysis.")
        # Generate a minimal report indicating load failure
        analyzer.report_data['total_rows_initially_loaded'] = 0
        analyzer.report_data['rows_after_preprocessing'] = 0
        analyzer.report_data['domain_analysis'] = "--- Domain Analysis ---\n  Analysis skipped: Data loading failed or file is empty."
        analyzer.report_data['sender_analysis'] = "--- Sender Analysis ---\n  Analysis skipped: Data loading failed or file is empty."
        analyzer.report_data['username_pattern_analysis'] = "--- Username Pattern Analysis ---\n  Analysis skipped: Data loading failed or file is empty."
        analyzer.report_data['action_language_analysis'] = "--- Action Language Analysis (Subjects) ---\n  Analysis skipped: Data loading failed or file is empty."

        analyzer.generate_report()
        # Save report in the same directory as the input file
        output_dir = os.path.dirname(args.filepath)
        if not output_dir: output_dir = '.'
        report_save_path = os.path.join(output_dir, REPORT_FILENAME)
        analyzer.save_report(report_save_path)
        sys.exit(1) # Exit with error code


    # 2. Perform Analysis (including ML and pattern analysis)
    # perform_analysis returns True if ML steps for viz succeeded and data exists, False otherwise
    ml_steps_for_viz_successful = analyzer.perform_analysis()


    # 3. Generate Visualization (only if ML analysis was successful)
    visualization_generated = False
    if ml_steps_for_viz_successful:
         visualization_generated = analyzer.generate_visualization()
    else:
         print("\nML analysis steps required for visualization did not complete successfully or yielded no data. Skipping visualization generation.")


    # 4. Generate Report (compiles findings regardless of ML success)
    # The report will reflect whether ML analysis and visualization were performed
    analyzer.generate_report()

    # 5. Save Report
    # Save report in the same directory as the input file
    output_dir = os.path.dirname(args.filepath)
    if not output_dir:
         output_dir = '.' # Current directory
    report_save_path = os.path.join(output_dir, REPORT_FILENAME)

    analyzer.save_report(report_save_path)

    # Optional: Inform user about visualization save path if generated
    if visualization_generated:
         plot_save_path = os.path.join(output_dir, CLUSTER_PLOT_FILENAME)
         print(f"\nVisualization file: {plot_save_path}")
    else:
         print(f"\nVisualization file was not generated.")


if __name__ == "__main__":
    main()