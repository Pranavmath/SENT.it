from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import json
import os.path
import sys

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

def return_soup(URL):
  driver = webdriver.Chrome(options=chrome_options)
  driver.get(URL)
  return BeautifulSoup(driver.page_source, features="html5lib")

MAIN_URL = "https://www.malacards.org"

soup = return_soup("https://www.malacards.org/categories/ear_disease_list")

table = soup.find("table", {"class": "search-results"})
rows = table.find_all("tr")

best_ear = {}

for row_n in range(len(rows)):
  disease = rows[row_n].find("a")
  disease_name = disease.text

  CARD_URL = MAIN_URL + disease["href"]

  print(CARD_URL)

  symp_soup = return_soup(CARD_URL)

  clinical_features = symp_soup.find("div", {"id": "clinical_features"})

  spans = clinical_features.find_all("span", {"itemprop": None, "class": None})

  #print(spans)

  symptoms = []

  for span in spans:
    span_p = span.find("p")
    if span_p and span_p.text:
      symptoms.append(span_p.text.strip())
  
  best_ear[disease_name] = symptoms

with open("best_ear.json", "w") as f:
  json.dump(best_ear, f)