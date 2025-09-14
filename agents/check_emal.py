
# Sender and receiver
sender_email = "lalithashreecp@gmail.com"
receiver_email = "lalithashreelalli@gmail.com"
app_password = "uqee mmqg dbgv anrl"  # Use App Password if 2FA is on

# Email content
message = MIMEMultipart("alternative")
message["Subject"] = "Test Email via SMTP"
message["From"] = sender_email
message["To"] = receiver_email

# Plain text and HTML versions
text = "Hello,\nThis is a plain text email sent via Python!"
html = """\
<html>
  <body>
    <p>Hello,<br>
       This is an <b>HTML</b> email sent via Python!
    </p>
  </body>
</html>
"""

# Attach both
message.attach(MIMEText(text, "plain"))
message.attach(MIMEText(html, "html"))

# Sending email
try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, message.as_string())
    print("✅ Email sent successfully!")
except Exception as e:
    print("❌ Failed to send email:")
    traceback.print_exc()
