# api to get bitcoin prices
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0"
}

# url = input("Enter URL:").strip()
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90"

response = requests.get(url, headers=headers)
print(response.status_code)
print(response.json())
