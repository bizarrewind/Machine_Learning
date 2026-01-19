from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:145.0) Gecko/20100101 Firefox/145.0"
}

url = "https://www.flipkart.com/samsung-galaxy-s24-5g-snapdragon-onyx-black-128-gb/p/itm3469a7107606f?pid=MOBHDVFKSSHPUYHB"
# url = input("Enter URL:").strip()

response = requests.get(url, headers=headers)
print(response.status_code)
print(response.text)
