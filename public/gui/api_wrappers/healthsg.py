import requests

def get_healthsage_data(params):
    url = "https://healthsage.ai/api/SomeEndpoint"
    response = requests.get(url, params=params)
    return response.json()
