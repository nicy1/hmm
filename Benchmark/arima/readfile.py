import json
import csv
from collections import OrderedDict



# ==========================================================================


class reader:
    def __init__ (self, filename):
        self.filename = filename

    def read_csv (self): 
        table = {"2":0.0, "4":1.0, "5":2.0}  
        series = []
        data = csv.DictReader(open(self.filename, mode='r'))
        for row in data:
          x = str(row['ActionType'])
          series.append(table[x])

        return series 

# ==========================================================================