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
    if i[:3] == "BTC":
        b.append(i)
    if i[:3] == "ETH":
        e.append(i)
    if i[:3] == "XRP":
        x.append(i)
        
df = zip([b, e, x], ["BTC", "ETH", "XRP"])

for ls, name in df:
    summ = 0
    for bb in ls:
        summ+=count_lines(DATA_FOLDER + bb)
    print(name, len(ls), "days", summ, "rows")