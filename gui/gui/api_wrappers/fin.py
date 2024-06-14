# import requests

# def get_financial_data(params):
#     url = "https://finchat.io/api/SomeEndpoint"
#     response = requests.get(url, params=params)
#     return response.json()


import investpy

def get_financial_data(params):
    # Example parameters: 'ticker' is assumed to be in params
    ticker = params.get('ticker', '')
    
    # Example of using investpy to get historical data
    historical_data = investpy.get_stock_historical_data(stock=ticker,
                                                        country='United States',
                                                        from_date='01/01/2021',
                                                        to_date='31/12/2021')
    
    return historical_data
