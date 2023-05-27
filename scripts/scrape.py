import twint
import pandas as pd
import datetime
import sys

def count_lines(filename):
    with open(filename) as fp:
        count = 0
        for _ in fp:
            count += 1
    return count
        
TICKER = sys.argv[1]

c = twint.Config()
c.Search = '$' + TICKER
c.Hide_output = True
c.Store_csv = True
c.Custom_csv = ["language", "created_at", "tweet", "username"]

start_date = datetime.date(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
end_date   = datetime.date(2023, 2, 19)

date_generated = [ str(start_date + datetime.timedelta(n)) for n in range(int ((end_date - start_date).days))]

for i in range(len(date_generated) - 1):
    c.Since = date_generated[i]
    c.Until = date_generated[i+1]
    filename = "scraped_data/" + TICKER + date_generated[i] + ".csv"
    c.Output =  filename
    twint.run.Search(c)
    tlist = c.search_tweet_list
    print(len(tlist))
    print(date_generated[i], count_lines(filename))
    