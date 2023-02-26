import pandas as pd
import os

DATA_FOLDER = "scraped_data/"

def count_lines(filename):
    with open(filename) as fp:
        count = 0
        for _ in fp:
            count += 1
    return count

b = []
e = []
x = []
for i in os.listdir(DATA_FOLDER):
    prefix = i[:3]
    if prefix == "BTC":
        b.append(i)
    if prefix == "ETH":
        e.append(i)
    if prefix == "XRP":
        x.append(i)
        
df = zip([b, e, x], ["BTC", "ETH", "XRP"])

for files, name in df:
    rowcount = 0
    for file in files:
        rowcount+=count_lines(DATA_FOLDER + file)
    print(name, len(files), "days", rowcount, "rows")