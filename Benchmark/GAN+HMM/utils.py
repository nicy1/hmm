import sys
import os
import math

# convert predicted values into real data format

class Utils():
      def __init__(self):
          self.y_lookUpTable = {}


      def convert(self, predictions):

          for i in range(len(predictions)):
              y = predictions[i]
              if y not in self.y_lookUpTable:
                 self.y_lookUpTable[y] = len(self.y_lookUpTable)

              predictions[i] = self.y_lookUpTable[y]

          return predictions
