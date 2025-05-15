import pandas as pd
import numpy as np
import email.utils
import re
import unicodedata
from datetime import datetime
import os

# Constants
COLUMN_MAPPING = {
    "sender": "Sender",
    "receiver": "Recipient",
    "date": "EmailDate",
    "subject": "Subject",
    "body": "EmailBody",
    "urls": "URLs",
    "label": "Label"
}
OUTPUT_COLUMNS = ["SenderName", "SenderEmail", "SenderDomain", "RecipientName", "RecipientEmail", "EmailDate", "Subject", "EmailBody", "URLs", "Label"]

# Utility Functions
def clean_text(text):
    if pd.isnull(text) or not isinstance(text, str):
        return "NULL"
    text = unicodedata.normalize('NFKC', str(text))
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def extract_domain(email):
    if pd.isnull(email) or '@' not in email:
        return "NULL"
    return email.split('@')[-1]

def is_junk_text(text):
    if pd.isnull(text):
        return True
    patterns = [r'A{3,}fA{3,}', r'A{4,}fA\?s', r'(.)\1{4,}']
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

# Email Data Processor Class
class EmailDataProcessor:
    def __init__(self, input_file, encoding):
        self.input_file = input_file
        self.encoding = encoding
        self.data = None
        self.junk_rows = 0
        self.total_rows = 0
        self.null_counts = {}

    def load_file(self):
        try:
            if self.input_file.endswith('.csv'):
                self.data = pd.read_csv(self.input_file, encoding=self.encoding)
            elif self.input_file.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(self.input_file, engine='openpyxl')
            else:
                raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")
        except Exception as e:
            print(f"Error loading file: {e}")
            exit(1)

        self.data.rename(columns=COLUMN_MAPPING, inplace=True, errors='ignore')
        self.total_rows = len(self.data)

    def extract_email_components(self):
        self.data['SenderName'], self.data['SenderEmail'] = zip(*self.data['Sender'].apply(self._parse_email))
        self.data['RecipientName'], self.data['RecipientEmail'] = zip(*self.data['Recipient'].apply(self._parse_email))
        self.data['SenderDomain'] = self.data['SenderEmail'].apply(extract_domain)

    def _parse_email(self, email_str):
        if pd.isnull(email_str):
            return "NULL", "NULL"
        parsed = email.utils.parseaddr(email_str)
        name = clean_text(parsed[0]) or clean_text(parsed[1].split('@')[0])
        email_addr = parsed[1] if '@' in parsed[1] else "NULL"
        return name, email_addr

    def parse_email_date(self):
        def normalize_date(date_str):
            try:
                return pd.to_datetime(date_str, errors='coerce').strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                return 'NULL'
        self.data['EmailDate'] = self.data['EmailDate'].apply(normalize_date)

    def is_junk_row(self, row):
        null_count = row.isnull().sum() + (row == 'NULL').sum()
        if null_count > len(row) // 2:
            return True
        if any(is_junk_text(str(value)) for value in row):
            return True
        return False

    def process_data(self):
        self.data = self.data.fillna('NULL')
        self.data['EmailBody'] = self.data['EmailBody'].apply(lambda x: clean_text(x)[:1000])

        self.extract_email_components()
        self.parse_email_date()

        is_junk = self.data.apply(self.is_junk_row, axis=1)
        self.junk_rows = is_junk.sum()
        self.data = self.data[~is_junk].reset_index(drop=True)

        self.null_counts = self.data.isin(['NULL']).sum().to_dict()

    def generate_report(self):
        report = []
        report.append(f"Total rows initially loaded: {self.total_rows}")
        report.append(f"Rows removed as junk: {self.junk_rows}")
        report.append("NULL value counts and percentages:")
        for col, count in self.null_counts.items():
            percentage = (count / len(self.data)) * 100 if len(self.data) > 0 else 0
            report.append(f"  {col}: {count} ({percentage:.2f}%)")
        report.append("Output columns:")
        report.append(", ".join(OUTPUT_COLUMNS))
        report.append("Sample of processed data (first 5 rows):")
        report.append(self.data.head().to_string())
        return "\n".join(report)

    def save_report(self, report):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"processing_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_file}")

    def save_processed_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"processed_email_data_{timestamp}.csv"
        self.data.to_csv(output_file, index=False, columns=OUTPUT_COLUMNS)
        print(f"Processed data saved to {output_file}")

def main():
    # Interactive prompts for file path and encoding
    print("Welcome to the Email Classifier!")
    input_file = input("Please provide the path to the input file (CSV or Excel): ").strip()

    while not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        input_file = input("Please provide a valid file path: ").strip()

    encoding = input("Please provide the file encoding (default is 'utf-8'): ").strip()
    if not encoding:
        encoding = "utf-8"

    processor = EmailDataProcessor(input_file, encoding)
    processor.load_file()
    processor.process_data()

    report = processor.generate_report()
    print("\nProcessing Report:\n")
    print(report)

    save = input("\nDo you want to save the processed data and report? (yes/no): ").strip().lower()
    if save == 'yes':
        processor.save_report(report)
        processor.save_processed_data()

if __name__ == "__main__":
    main()