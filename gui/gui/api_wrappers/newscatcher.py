import requests

def get_news_data(params):
    url = "https://api.newscatcherapi.com/v2/SomeEndpoint"
    headers = {"x-api-key": "YOUR_API_KEY"}
    response = requests.get(url, headers=headers, params=params)
    return response.json()


# pip install pytrends
# pytrends.get_historical_interest(kw_list, year_start=2018, month_start=1, day_start=1, hour_start=0, year_end=2018, month_end=2, day_end=1, hour_end=0, cat=0, geo='', gprop='', sleep=0)