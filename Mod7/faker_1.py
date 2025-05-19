import csv
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Common email domains
email_domains = [
    "gmail.com", "outlook.com", "yahoo.com", "hotmail.com",
    "protonmail.com", "ymail.com", "msn.com"
]

# 25 subject templates with placeholders
subject_templates = [
    "Reminder: Your appointment on {day_of_week}",
    "Upcoming {event} this {day_of_week}",
    "Meeting scheduled with {company} on {date}",
    "Don't miss the {announcement} from {company}",
    "Update: {project} milestone on {date}",
    "Re: Discussion on {topic} scheduled {meeting_time}",
    "Invitation to {event} by {company}",
    "Action Required: {topic} review before {date}",
    "Monthly summary of {project} - {month}",
    "Join us for the {announcement} this {day_of_week}",
    "New updates about {product} from {company}",
    "You're invited: {event} on {date}",
    "Final call for {topic} session on {date}",
    "Your feedback needed for {project}",
    "Re: Appointment confirmation for {date}",
    "Important: {announcement} update from {company}",
    "Let's discuss {topic} at {meeting_time}",
    "Quick reminder: {event} on {month} {date}",
    "See you at the {event}!",
    "Agenda for upcoming {project} review",
    "Happy to announce {announcement} from {company}",
    "Can we meet to talk about {project}?",
    "Next steps for {topic} initiative",
    "Schedule your session on {date}",
    "Details regarding the {event} you registered for"
]

# 40 body templates with placeholders (single line)
body_templates = [
    "Hi {first_name}, just a quick note from our team at {company} about your {product} appointment on {date} in {city}.",
    "Hello {first_name}, we're looking forward to seeing you in {city} this {month} at {time}.",
    "Hi {first_name}, your {product} has been successfully scheduled for delivery to {city} on {date}.",
    "Dear {first_name}, thank you for contacting {company}. We'll reach out again by {date}.",
    "Greetings {first_name}, this is to inform you of your updated {product} status from {company} as of {date}.",
    "Hi {first_name}, we hope you're enjoying your experience with {product}. Let us know how we did!",
    "Hello {first_name}, your order from {company} will arrive in {city} on {date}.",
    "Dear {first_name}, we appreciate your loyalty to {company}. Expect an update by {time} tomorrow.",
    "Hi {first_name}, just confirming your upcoming meeting at {company} in {city} this {month}.",
    "Hello {first_name}, the requested documents will be available by {time} on {date}.",
    "Hi {first_name}, please confirm your availability for our meeting on {date} at {time}.",
    "Dear {first_name}, we've updated your records in our system as of {date}.",
    "Hello {first_name}, a reminder about your scheduled check-in with {company} this {month}.",
    "Hi {first_name}, please review the attached summary of your {product} scheduled for {date}.",
    "Dear {first_name}, we noticed a recent activity from {city} on your {product}.",
    "Hi {first_name}, we’re excited to see you in {city} this {month} for the {product} workshop.",
    "Hello {first_name}, your {product} evaluation is complete. Check your inbox on {date}.",
    "Dear {first_name}, we’ve received your form and will contact you by {time} on {date}.",
    "Hi {first_name}, we're confirming your attendance to our {product} training in {city}.",
    "Hello {first_name}, your {product} account has been reviewed and updated as of {date}.",
    "Dear {first_name}, we've assigned your ticket to our {company} representative in {city}.",
    "Hi {first_name}, your next billing date is set for {date}.",
    "Hello {first_name}, please check your calendar for availability in {month}.",
    "Hi {first_name}, your scheduled call with {company} is on {date} at {time}.",
    "Dear {first_name}, thanks for participating in our {product} beta in {city}.",
    "Hi {first_name}, we're happy to announce a new upgrade to {product}, launching on {date}.",
    "Hello {first_name}, your password reset link was sent at {time} today.",
    "Dear {first_name}, please join us at {company}'s online session on {date}.",
    "Hi {first_name}, just checking in regarding your experience with {product} in {city}.",
    "Hello {first_name}, your monthly report from {company} is now available.",
    "Dear {first_name}, thank you for choosing {company}. Your service will activate on {date}.",
    "Hi {first_name}, we've attached the agenda for the upcoming call with {company}.",
    "Hello {first_name}, please download your receipt from {company} dated {date}.",
    "Dear {first_name}, here’s a summary of your activity in {month} on {product}.",
    "Hi {first_name}, your scheduled visit to {company} is confirmed for {date}.",
    "Hello {first_name}, our team will be in touch by {time} on {date}.",
    "Dear {first_name}, we’re preparing your shipment to {city} and it should arrive by {date}.",
    "Hi {first_name}, we hope to see you at our event in {city} on {date}.",
    "Hello {first_name}, your trial for {product} has been extended through {month}.",
    "Dear {first_name}, don't forget your appointment at {time} this {month}."
]

# Function to generate a random email address
def generate_email():
    name = fake.first_name().lower() + "." + fake.last_name().lower()
    domain = random.choice(email_domains)
    return f"{name}@{domain}"

# Function to fill placeholders in a template
def fill_template(template):
    return template.format(
        first_name=fake.first_name(),
        city=fake.city(),
        company=fake.company(),
        product=fake.bs().split()[0].capitalize(),
        date=fake.date_this_year().strftime("%Y-%m-%d"),
        month=fake.month_name(),
        time=fake.time(),
        day_of_week=fake.day_of_week(),
        event=fake.catch_phrase(),
        topic=fake.word().capitalize(),
        announcement=fake.bs().capitalize(),
        meeting_time=fake.time(),
        appointment=fake.date_this_year().strftime("%Y-%m-%d"),
        project=fake.bs().capitalize()
    )

# Generate 500 email records
records = []
for _ in range(500):
    sender_email = generate_email()
    recipient_email = generate_email()
    subject_template = random.choice(subject_templates)
    body_template = random.choice(body_templates)
    subject = fill_template(subject_template)
    body = fill_template(body_template).replace('\n', ' ').replace('\r', ' ')
    records.append([sender_email, recipient_email, subject, body, "good"])

# Write records to CSV file
with open("varied_emails.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["sender_email", "recipient_email", "subject", "body", "label"])
    writer.writerows(records)

print("CSV file 'varied_emails.csv' with 500 email records has been successfully created.")
