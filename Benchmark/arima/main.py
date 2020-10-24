import readfile as rf

from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

# ==========================================================================

index_table = {0.0:'0', 1.0:'1', 2.0:'2', 3.0:'4', 4.0:'5'}
count = 0

def proper_round(num, dec=0):
  num = str(num)[:str(num).index('.')+dec+2]
  if num[-1]>='5':
    return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
  return float(num[:-1])

reader = rf.reader('telepathology2.csv')
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
   
accuracy = accuracy_score(test, predictions)
precision, recall, fscore, support = score(test, predictions)
print('accuracy = %.2f' % (accuracy*100), '%')
print('Precision:', precision)
print('Recall:', recall)
print('F-Measure:', fscore)
print('Support-score:', support)


# ==========================================================================


