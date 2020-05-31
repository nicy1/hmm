# hmm
Implementation of Hidden Markov Model (HMM) using python3

File 'hmm.py' (Main file)
-------------------------
HMM's Functions:
 - forward():  computes the likelihood P(O|λ)
 - backward(): it is the time-reversed version of the Forward Algorithm
 - viterbi():  computes the best hidden state sequence Q (Decoding)
 - forward_backward(): Learn the HMM parameters A, B and Pi (Learning)
 
File 'localdataset.json' 
------------------------
It is a dataset created for testing purpose.

Contents:
 - States: Rainy, Sunny
 - Observations: Walk, Shop, Clean
 - A (states transitions probabilities matrix), B (states emissions probabilities matrix), pi (states start prob.)
  
File 'death_recovered.csv'
--------------------------
This dataset illustrates the COVID-19 stats and patients tracking in India, at the state and city level, over time.

It is just a part of the complete dataset 'death_and_recovered.csv' (main one used) located at the following link: https://api.covid19india.org/csv/latest/death_and_recovered.csv

Contents:
 - Sl_No: serial number of the record
 - Date: record's date
 - Age Bracket: age of the patient
 - Gender: gender of the patient
 - Patient_Status: deceased/recovered
 - City: city where stay the patient
 - District: district where stay the patient
 - State:  state where stay the patient
 - Statecode: state code
 - Notes: some informations about the patient
 - Nationality: of patient
 - Source_1: source of the information about the patient
 - Source_2:
 - Source_3:
 - Patient_Number (Could be mapped later):

