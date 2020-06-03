import sys
import hmm
import requests
import csv
import json



#path = 'https://github.com/nicy1/hmm/blob/master/dataset/death_recovered.csv'
#path = 'https://github.com/nicy1/hmm/blob/master/dataset/touch_dynamics.csv'
#text = requests.get(path).text
#lines = text.splitlines()
#reader = csv.DictReader(lines)   
    
#extract the name of file
#firstpos = path.rfind("/")
#lastpos  = path.rfind(".")
#self.filename = path[firstpos+1 : lastpos]
#self.filename += '.json'





#get dataset in .csv and convert it in .json 
filename = 'death_recovered'
csvfile = open(filename + '.csv', 'r')
reader = csv.DictReader(csvfile)

jsonfile = open(filename + '.json', 'w+')
d = []
for r in reader:
  d.append(r)
jsonfile.write(json.dumps(d, indent=4))


#test starts
choice = int (input('Choose local(0) or external(1) dataset: '))
obs = []

if choice == 0:                                #TEST WITH MY LOCAL DATASET
  h = hmm.Hmm('localdataset.json')

  #create a sequence of observations for testing
  obs = ['Walk', 'Shop', 'Clean']

else:                                         #TEST FOR THE TELEMEDICINE OBSERVATIONS DATASET
  h = hmm.Hmm(filename + '.json')
  
  data = json.loads(open(filename+'.json').read())
  for record in data:
    obs.append(record['Patient_Status'])
 
  

# test forward algorithm
print (h.forward(obs))
print ('----------------------------\n')
# test backward algorithm
print (h.backward(obs))
print ('----------------------------\n')
# test viterbi algorithm
print (h.viterbi(obs))
print ('----------------------------\n')

# test forward_backward algorithm
A, B, pi = h.forward_backward(obs)
print ('\nTRANSITION PROBABILITIES--> ', A)
print ('\nEMISSION PROBABILITIES--> ', B)
print ('\nINITIAL PROBABILITIES--> ', pi)



  
  
