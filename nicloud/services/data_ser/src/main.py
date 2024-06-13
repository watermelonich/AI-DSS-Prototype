# services/data_ser/src/main.py
from flask import Flask, request, jsonify
from config.settings import Config
from models.data_model import db, DataModel, DataModelSchema
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

@app.errorhandler(ValidationError)
def handle_validation_error(e):
    response = jsonify(e.messages)
    response.status_code = 400
    return response

@app.route('/data', methods=['POST'])
def add_data():
    try:
        data = request.get_json()
        schema = DataModelSchema()
        result = schema.load(data)
        new_data = DataModel(name=result['name'], value=result['value'])
        db.session.add(new_data)
        db.session.commit()
        return jsonify(schema.dump(new_data)), 201
    except ValidationError as err:
        return jsonify(err.messages), 400
    except Exception as err:
        app.logger.error(f"Error adding data: {err}")
        return jsonify({"message": "Internal Server Error"}), 500

@app.route('/data', methods=['GET'])
def get_data():
    try:
        data = DataModel.query.all()
        schema = DataModelSchema(many=True)
        return jsonify(schema.dump(data)), 200
    except Exception as err:
        app.logger.error(f"Error fetching data: {err}")
        return jsonify({"message": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
