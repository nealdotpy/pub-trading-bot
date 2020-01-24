import smtplib, ssl
import time, calendar

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from google.cloud import storage

class emailserver:
    PORT = 465 # SSL port
    password = ''
    sender_email = 'neal.rpi@gmail.com'
    smtp_server = 'smtp.gmail.com'
    email = MIMEMultipart()

    def __init__(self):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket('neal-trading-bot-bucket')
        self.password = bucket.blob('.email_password').download_as_string().decode()[:-1]

    def send(self, who, subject, body):
        self.email["From"] = self.sender_email
        self.email["To"] = who
        self.email["Subject"] = subject
        self.email.attach(MIMEText(body, "plain"))
        text = self.email.as_string()
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", self.PORT, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(self.sender_email, who, text)
        log = 'Sent an email from {} to {}:\nSubject: {}\n\nBody:\n{}'.format(
                self.sender_email, who, subject, body)
        print('{}\n\n\n'.format(log))