# services/notif/controllers/notif_controller.py
from flask import request, jsonify
from models.notif_model import db, Notification, NotificationSchema
from marshmallow import ValidationError
import logging

logger = logging.getLogger(__name__)

def add_notification():
    try:
        data = request.get_json()
        schema = NotificationSchema()
        result = schema.load(data)
        new_notification = Notification(**result)
        db.session.add(new_notification)
        db.session.commit()
        logger.info("Notification added successfully")
        return jsonify(schema.dump(new_notification)), 201
    except ValidationError as err:
        logger.error(f"Validation error: {err.messages}")
        return jsonify(err.messages), 400
    except Exception as err:
        logger.error(f"Error adding notification: {err}")
        return jsonify({"message": "Internal Server Error"}), 500

def get_notifications():
    try:
        notifications = Notification.query.all()
        schema = NotificationSchema(many=True)
        return jsonify(schema.dump(notifications)), 200
    except Exception as err:
        logger.error(f"Error fetching notifications: {err}")
        return jsonify({"message": "Internal Server Error"}), 500
