# web_app/forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, FileField, SubmitField
from wtforms.validators import Optional, URL

class UploadForm(FlaskForm):
    video_file = FileField("Video File", validators=[Optional()])
    video_url = StringField("Video URL (RTSP/HTTP)", validators=[Optional(), URL(message="Enter valid URL")])
    submit = SubmitField("Start")
