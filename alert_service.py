"""
SMART ALERT SERVICE
Handles: SMS + Email + Image Attachment for emergency alerts
"""

import os
from datetime import datetime
from twilio.rest import Client

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


# ---------------------------------------------------------------
# 🔐 TWILIO SMS SETTINGS
# ---------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', )
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', )
TWILIO_FROM_NUMBER = os.environ.get('TWILIO_FROM_NUMBER', )
DEFAULT_SMS_TO = os.environ.get('ALERT_TO_NUMBER', ')

sms_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# ---------------------------------------------------------------
# 🔐 EMAIL SETTINGS
# ---------------------------------------------------------------
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "r.sanjairsk@gmail.com")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD", "hxww kkov aqlo vhye")
DEFAULT_EMAIL_TO = os.environ.get("EMAIL_TO", "r.sanjairsk@gmail.com")


# ---------------------------------------------------------------
# 📧 SEND EMAIL ALERT (supports image attachments)
# ---------------------------------------------------------------
def send_email_alert(subject, message, to_email=None, image_bytes=None, image_filename=None):
    to_email = to_email if to_email else DEFAULT_EMAIL_TO

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg["Subject"] = subject

    # Email body
    msg.attach(MIMEText(message, "plain"))

    # Attach image if provided
    if image_bytes and image_filename:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(image_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={image_filename}")
        msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_SENDER, to_email, msg.as_string())
        server.quit()
        print("[EMAIL] Email sent successfully")
        return True
    except Exception as e:
        print("[EMAIL] Failed:", e)
        return False


# ---------------------------------------------------------------
# 📱 SEND SMS ALERT
# ---------------------------------------------------------------
def send_sms_alert(text, to_number=None):
    to_number = to_number if to_number else DEFAULT_SMS_TO

    try:
        message = sms_client.messages.create(
            body=text,
            from_=TWILIO_FROM_NUMBER,
            to=to_number
        )
        print("[SMS] SMS sent:", message.sid)
        return True
    except Exception as e:
        print("[SMS] Failed:", e)
        return False


# ---------------------------------------------------------------
# 🚨 MASTER ALERT FUNCTION — sends SMS + Email together
# ---------------------------------------------------------------
def send_alert(incident_type, location="THALAVAPALAYAM", confidence=0.0,
               image_bytes=None, image_filename=None, send_email=True):

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conf_pct = int(confidence * 100)

    alert_msg = (
        f"🚨 SMART ALERT 🚨\n"
        f"Type: {incident_type.upper()}\n"
        f"Location: {location}\n"
        f"Date/Time: {timestamp}\n"
        f"Confidence: {conf_pct}%\n"
        f"Status: ALERT TRIGGERED"
    )

    # ---- SMS ----
    sms_ok = send_sms_alert(alert_msg)

    # ---- EMAIL ----
    email_ok = False
    if send_email:
        email_ok = send_email_alert(
            subject=f"[SMART ALERT] {incident_type.upper()} DETECTED",
            message=alert_msg,
            image_bytes=image_bytes,
            image_filename=image_filename
        )

    return {
        "sms_sent": sms_ok,
        "email_sent": email_ok,
        "alert_text": alert_msg
    }


# ---------------------------------------------------------------
# Optional: Bulk alert (not required for single-frame emergency)
# ---------------------------------------------------------------
def send_bulk_alert(incidents: list):
    results = []
    for inc in incidents:
        result = send_alert(
            incident_type=inc.get("task", "unknown"),
            location=inc.get("location", "THALAVAPALAYAM"),
            confidence=inc.get("confidence", 0.0)
        )
        results.append(result)

    return {
        "total_incidents": len(incidents),
        "alerts_sent": sum(1 for r in results if r["sms_sent"] or r["email_sent"]),
        "details":results
    }