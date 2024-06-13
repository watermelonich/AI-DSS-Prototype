# services/notif/notif_model.py
from flask_sqlalchemy import SQLAlchemy
from marshmallow import Schema, fields, validate, ValidationError

db = SQLAlchemy()

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(255), nullable=False)
    recipient = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

    def __repr__(self):
        return f'<Notification {self.message}>'

class NotificationSchema(Schema):
    id = fields.Int(dump_only=True)
    message = fields.Str(required=True, validate=validate.Length(min=1))
    recipient = fields.Str(required=True, validate=validate.Length(min=1))
    timestamp = fields.DateTime(required=True)
