# MIFTS is a metric for how much information there is for a disease card is => related to how common the disease is

from bs4 import BeautifulSoup
import json

with open("skin_MIFTS.json") as f:
  data = json.load(f)

a = sorted(data.items(), key=lambda item: item[1], reverse=True)

for t in a[:15]:
  print(t)


def run(file_path):
  diseases_MIFTS = {}

  with open(file_path) as fp:
    soup = BeautifulSoup(fp, "html.parser")

  table = soup.find("table", {"class", "search-results"})

  rows = table.find_all("tr")[1:]

  for row in rows:
    disease = row.find("a", {
        "title": False,
        "class": False
    }).get_text().strip()
    MIFTS = row.find("td", {
        "title":
        "MalaCards InFormaTion Score- annotation strength (max 100)"
    }).get_text().strip()

    diseases_MIFTS[disease] = int(MIFTS)

  return diseases_MIFTS


def disease_dump(type_disease):
  with open(f"{type_disease}_MIFTS.json", "w") as f:
    disease_dict = run(f"{type_disease}.html")
    json.dump(disease_dict, f)


disease_dump("ear")
disease_dump("nose")
disease_dump("oral")
disease_dump("skin")
