import readfile as rf

from matplotlib import pyplot
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# ==========================================================================

index_table = {0.0:'2', 1.0:'4', 2.0:'5'}
count = 0

def proper_round(num, dec=0):
  num = str(num)[:str(num).index('.')+dec+2]
  if num[-1]>='5':
    return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
  return float(num[:-1])

reader = rf.reader('telepathology.csv')
X = reader.read_csv()
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
   model = ARIMA(history, order=(1,1,0))
   model_fit = model.fit(disp=0)
   output = model_fit.forecast()
   yhat = proper_round(output[0][0])
   predictions.append(yhat)
   obs = test[t]
   history.append(obs)
   
   if index_table[yhat] == index_table[obs]:
     count += 1
   print('predicted=%s, expected=%s' % (index_table[yhat], index_table[obs]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

acc = (count/len(test)) * 100
print('accuracy = %.2f' % acc, '%')


# ==========================================================================


