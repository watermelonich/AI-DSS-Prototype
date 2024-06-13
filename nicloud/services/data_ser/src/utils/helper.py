# services/data_ser/src/utils/helper.py
def format_response(data):
    return {
        'id': data.id,
        'name': data.name,
        'value': data.value
    }
