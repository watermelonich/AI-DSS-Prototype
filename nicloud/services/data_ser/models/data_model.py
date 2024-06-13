# services/data_ser/models/data_model.py
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import validates
from marshmallow import Schema, fields, validate, ValidationError

db = SQLAlchemy()

class DataModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    value = db.Column(db.Integer, nullable=False)

    @validates('name')
    def validate_name(self, key, value):
        if not value:
            raise ValueError("Name cannot be empty")
        return value

    @validates('value')
    def validate_value(self, key, value):
        if value < 0:
            raise ValueError("Value cannot be negative")
        return value

    def __repr__(self):
        return f'<DataModel {self.name}>'

class DataModelSchema(Schema):
    id = fields.Int(dump_only=True)
    name = fields.Str(required=True, validate=validate.Length(min=1))
    value = fields.Int(required=True, validate=validate.Range(min=0))
