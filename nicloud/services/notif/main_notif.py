# services/notif/main_notif.py
from flask import Flask
from config.notif_settings import Config
from models.notif_model import db
from controllers.notif_controller import add_notification, get_notifications
import logging
from logging.config import dictConfig

# Configure logging
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': Config.LOG_LEVEL,
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/notifications', methods=['POST'])
def add_notification_route():
    return add_notification()

@app.route('/notifications', methods=['GET'])
def get_notifications_route():
    return get_notifications()

if __name__ == '__main__':
    app.run(debug=True)
