# Main file
import csv

with open("./data/fake_and_real_news.csv", mode='r', encoding='utf-8') as file:
    data = csv.reader(file)
    # [Text, label]
    for row in data:
        print(row[1])

