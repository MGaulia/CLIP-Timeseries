import sys
import os
from datetime import  datetime

TEST = "TEST"

def count_lines(filename):
    with open(filename) as fp:
        count = 0
        for _ in fp:
            count += 1
    return count

SCRAPED_DIR = "scraped_data"

files = os.listdir(SCRAPED_DIR)

d = {}

for f in files:
    ticker = f[:3]
    if ticker in d:
        temp = d[ticker]
        temp.append(f[3:-4])
        d[ticker] = temp
    else:
        d[ticker] = []
        
start_dates = {}
for ticker, files in d.items():
    files.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    start_dates[ticker] = files[-1]

for ticker, start_date in start_dates.items():
    start = start_date.split("-")
    c = twint.Config()
    c.Search = '$' + ticker
    c.Hide_output = True
    c.Store_csv = True
    c.Custom_csv = ["language", "created_at", "tweet", "username"]

    start_date = datetime.date(start[0], start[1], start[2])
    end_date   = datetime.date(2023, 1, 1)

    date_generated = [ str(start_date + datetime.timedelta(n)) for n in range(int ((end_date - start_date).days))]

    for i in range(len(date_generated) - 1):
        c.Since = date_generated[i]
        c.Until = date_generated[i+1]
        filename = "scraped_data/" + TICKER + date_generated[i] + ".csv"
        c.Output =  filename
        twint.run.Search(c)
        print(date_generated[i], count_lines(filename))
        