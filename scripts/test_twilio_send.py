import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())            # load .env into environment
from sms.twilio_adapter import send_sms_twilio

recipient = os.environ.get('NOTIFY_ADMIN_NUMBER') or os.environ.get('TWILIO_TEST_NUMBER') or '+919398294755'
print('Using recipient:', recipient)

ok = send_sms_twilio(recipient, 'Test SMS from Triple Riding Detector — ignore', mark_violation_id=None)
print('send_sms_twilio returned:', ok)
