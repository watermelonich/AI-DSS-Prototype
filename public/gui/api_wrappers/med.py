import requests

def get_medical_data(params):
    url = "https://endlessmedical.com/api/SomeEndpoint"
    response = requests.get(url, params=params)
    return response.json()
