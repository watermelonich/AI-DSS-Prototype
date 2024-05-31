import requests

def get_financial_data(params):
    url = "https://finchat.io/api/SomeEndpoint"
    response = requests.get(url, params=params)
    return response.json()
