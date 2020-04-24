import os
import pandas as pd
from collections import defaultdict
from collections import OrderedDict
import csv

data_path = os.getcwd()

train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))

class_image = defaultdict(list)
selected = train_wkt[train_wkt['MultipolygonWKT'] != 'MULTIPOLYGON EMPTY']
for i in range(len(selected)):
    class_image[selected.iloc[i, 1]].append(selected.iloc[i, 0])
class_image = OrderedDict(sorted(class_image.items()))

with open('class_image.csv', 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    for key, value in class_image.items():
       writer.writerow([key, len(value), value])