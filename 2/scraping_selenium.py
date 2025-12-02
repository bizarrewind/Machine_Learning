from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://www.amazon.in/HyperX-Cloud-Mini-PlayStation-Controllers/dp/B0D7NVR5SY/"

service = Service(GeckoDriverManager().install())

options = Options()
options.add_argument("--headless")
driver = webdriver.Firefox(service=service, options=options)

try:
    driver.get(URL)
    wait = WebDriverWait(driver, 10)
    html = driver.page_source
    print(f"Size of page = {len(html)} characters")
    print("_" * 20)
    print(html)


except Exception as e:
    print(f"An error occurred during scraping: {e}")

finally:
    driver.quit()
