import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_gmail(subject, body, recipient_email):
    gmail_user = 'your_email@gmail.com'  # Replace with your email
    gmail_password = 'your_password'  # Replace with your password or App Password
    
    # Email content
    message = MIMEMultipart()
    message['From'] = gmail_user
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        # Set up the SMTP server
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, recipient_email, message.as_string())
        server.close()
        print('Email sent!')
    except Exception as e:
        print(f'Failed to send email: {e}')

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_secure_email(subject, body, recipient_email, sender_email, sender_password):
    # Email content
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        # Set up the SMTP server with SSL
        # If using port 465 (SSL)
        server = smtplib.SMTP_SSL('mail92139.maychuemail.com', 465)
        # Or, if you want to use STARTTLS with port 587
        # server = smtplib.SMTP('mail92139.maychuemail.com', 587)
        # server.starttls()  # Upgrade the connection to secure
        
        # Log in to the SMTP server
        server.login(sender_email, sender_password)
        
        # Send the email
        server.sendmail(sender_email, recipient_email, message.as_string())
        
        # Close the connection to the server
        server.quit()
        
        print('Email sent successfully!')
    except Exception as e:
        print(f'Failed to send email: {e}')
