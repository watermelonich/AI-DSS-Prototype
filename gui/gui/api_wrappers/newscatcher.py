import requests

def get_news_data(params):
    url = "https://api.newscatcherapi.com/v2/SomeEndpoint"
    headers = {"x-api-key": "YOUR_API_KEY"}
    response = requests.get(url, headers=headers, params=params)
    return response.json()
