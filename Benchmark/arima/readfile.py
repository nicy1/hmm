import json
import csv
from collections import OrderedDict



# ==========================================================================


class reader:
    def __init__ (self, filename):
        self.filename = filename

    def read_csv (self):   
        series = []
        data = csv.DictReader(open(self.filename, mode='r'))
        for row in data:
          x = str(row['Protocol'])
          if x == 'TCP':
            series.append(1.0)
          elif x == 'TLSv1.2':
            series.append(2.0) 

            

        return series 

# ==========================================================================