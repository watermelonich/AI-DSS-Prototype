import requests

def get_supply_chain_data(params):
    url = "https://rapidapi.com/flowlogic-flowlogic/api/supply-chain-optimizer/SomeEndpoint"
    headers = {"x-rapidapi-key": "YOUR_RAPIDAPI_KEY"}
    response = requests.get(url, headers=headers, params=params)
    return response.json()
