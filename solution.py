# India hacks ml 2017: predict the segment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from collections import Counter


sample_sub = pd.read_csv('sample_submission.csv')
file_name_1 = "train_data.json"
with open(file_name_1, 'r') as jsonfile1:
    data_dict_1 = json.load(jsonfile1)
    
file_name_2 = "test_data.json"
with open(file_name_2, 'r') as jsonfile2:
    data_dict_2 = json.load(jsonfile2)

train = pd.DataFrame.from_dict(data_dict_1, orient='index')
train.reset_index(level=0, inplace=True)
train.rename(columns = {'index':'ID'},inplace=True)
print train.shape

test = pd.DataFrame.from_dict(data_dict_2, orient='index')
test.reset_index(level=0, inplace=True)
test.rename(columns = {'index':'ID'},inplace=True)
print test.shape

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)


test['segment'] = 'neg'
train_test = pd.concat([train, test])


train_test['segment'].loc[train_test['segment'] == 'neg'] = 0
train_test['segment'].loc[train_test['segment'] == 'pos'] = 1


cities = []
genres = []
for i, row in train_test.iterrows():
    d = row['cities'].split(',')
    g = row['genres'].split(',')
    
    for e in d:
        cities.append(e.split(':')[0])
    for e in g:
        genres.append(e.split(':')[0])
    
cities = list(set(cities))
genres = list(set(genres))


titles = []
for i, row in train_test.iterrows():
    if row['segment'] == 0:
        continue
    t = row['titles'].split(',')
    
    for e in t:
        titles.append(e.split(':')[0])
        
    if i%10000 == 0:
        print i


print len(cities)
print len(genres)


top_cities = ['hyderabad', 'Unknown', 'nagari', 'gandhinagar', 'dubayy', 'new delhi', 'mumbai', 'rawalpindi', 'bengaluru', 'lahore', 'kolkata', 'bhopal', 'chandigarh', 'gurgaon', 'chennai', 'prabhadevi', 'delhi', 'nagar', 'bangalore', 'ahmedabad', 'karachi', 'pune', 'navi mumbai', 'secunderabad', 'dhaka']
top_titles = ['Koffee With Karan',  'Yeh Rishta Kya Kehlata Hai',  'Ishqbaaaz',  'Ye Hai Mohabbatein',  'Chandra Nandni',  'Naamkarann',  'Pardes Mein Hai Mera Dil',  'Tanhaiyan',  'Dil Boley Oberoi',  'Jana Na Dil Se Door',  'Saath Nibhaana Saathiya',  'Mere Angne Mein',  'MS Dhoni',  'Nach Baliye',  'Ghulaam',  'Suhani Si Ek Ladki',  'Dil Hai Hindustani',  'Savdhaan India',  'Koi Laut Ke Aaya Hai',  'Sarabhai Vs Sarabhai']


count_city = []
count_genre = []
count_titles = []
count_days = []
count_hours = []

avg_per_city = []
avg_per_genre = []
avg_per_title = []
avg_per_day = []
avg_per_hour = []

Koffee_With_Karan = [0]*300000
Yeh_Rishta_Kya_Kehlata_Hai = [0]*300000
Ishqbaaaz = [0]*300000
Ye_Hai_Mohabbatein = [0]*300000
Chandra_Nandni = [0]*300000
Naamkarann = [0]*300000
Pardes_Mein_Hai_Mera_Dil = [0]*300000
Tanhaiyan = [0]*300000
Dil_Boley_Oberoi = [0]*300000
Jana_Na_Dil_Se_Door = [0]*300000
Saath_Nibhaana_Saathiya = [0]*300000
Mere_Angne_Mein = [0]*300000
MS_Dhoni = [0]*300000
Nach_Baliye = [0]*300000
Ghulaam = [0]*300000
Suhani_Si_Ek_Ladki = [0]*300000
Dil_Hai_Hindustani = [0]*300000
Savdhaan_India = [0]*300000
Koi_Laut_Ke_Aaya_Hai = [0]*300000
Sarabhai_Vs_Sarabhai = [0]*300000

hyderabad = [0]*300000
Unknown = [0]*300000
nagari = [0]*300000
gandhinagar = [0]*300000
dubayy = [0]*300000
new_delhi = [0]*300000
mumbai = [0]*300000
rawalpindi = [0]*300000
bengaluru = [0]*300000
lahore = [0]*300000
kolkata = [0]*300000
bhopal = [0]*300000
chandigarh = [0]*300000
gurgaon = [0]*300000
chennai = [0]*300000
prabhadevi = [0]*300000
delhi = [0]*300000
nagar = [0]*300000
bangalore = [0]*300000
ahmedabad = [0]*300000
karachi = [0]*300000
pune = [0]*300000
navi_mumbai = [0]*300000
secunderabad = [0]*300000
dhaka = [0]*300000

Travel = [0]*300000
Kabaddi = [0]*300000
Crime = [0]*300000
Romance = [0]*300000
LiveTV = [0]*300000
Hockey = [0]*300000
FormulaE = [0]*300000
Comedy = [0]*300000
Teen = [0]*300000
Cricket = [0]*300000
Mythology = [0]*300000
NA = [0]*300000
Horror = [0]*300000
Football = [0]*300000
Awards = [0]*300000
Science = [0]*300000
Tennis = [0]*300000
Thriller = [0]*300000
Boxing = [0]*300000
Wildlife = [0]*300000
Kids = [0]*300000
IndiaVsSa = [0]*300000
TalkShow = [0]*300000
TableTennis = [0]*300000
Volleyball = [0]*300000
Drama = [0]*300000
Action = [0]*300000
Athletics = [0]*300000
Reality = [0]*300000
Documentary = [0]*300000
Swimming = [0]*300000
Formula1 = [0]*300000
Family = [0]*300000
Badminton = [0]*300000
Sport = [0]*300000

d1 = [0]*300000
d2 = [0]*300000
d3 = [0]*300000
d4 = [0]*300000
d5 = [0]*300000
d6 = [0]*300000
d7 = [0]*300000

h0 = [0]*300000
h1 = [0]*300000
h2 = [0]*300000
h3 = [0]*300000
h4 = [0]*300000
h5 = [0]*300000
h6 = [0]*300000
h7 = [0]*300000
h8 = [0]*300000
h9 = [0]*300000
h10 = [0]*300000
h11 = [0]*300000
h12 = [0]*300000
h13 = [0]*300000
h14 = [0]*300000
h15 = [0]*300000
h16 = [0]*300000
h17 = [0]*300000
h18 = [0]*300000
h19 = [0]*300000
h20 = [0]*300000
h21 = [0]*300000
h22 = [0]*300000
h23 = [0]*300000


k = 0
tot_time = []
for i, row in train_test.iterrows():
    t_time = 0
    c = row['cities'].split(',')
    g = row['genres'].split(',')
    t = row['titles'].split(',')
    d = row['dow'].split(',')
    h = row['tod'].split(',')
    
    count_city.append(len(c))
    count_genre.append(len(g))
    count_titles.append(len(t))
    count_days.append(len(d))
    count_hours.append(len(h))
    
    for e in c:
        if str(e.split(':')[0]) in top_cities:
            if str(e.split(':')[0]) == 'new delhi':
                new_delhi[k] = int(e.split(':')[1])
                #new_delhi[k] = 1
            elif str(e.split(':')[0]) == 'navi mumbai':
                navi_mumbai[k] = int(e.split(':')[1])
                #navi_mumbai[k] = 1
            else:
                eval(e.split(':')[0])[k] = int(e.split(':')[1])
                #eval(e.split(':')[0])[k] = 1
        
    for e in g:
        if str(e.split(':')[0]) == 'Table Tennis':
            TableTennis[k] = int(e.split(':')[1])
            #TableTennis[k] = 1
        else:
            eval(e.split(':')[0])[k] = int(e.split(':')[1])
            #eval(e.split(':')[0])[k] = 1
        
    for e in d:
        eval('d' + str(e.split(':')[0]))[k] = int(e.split(':')[1])
        #eval('d' + str(e.split(':')[0]))[k] = 1
        t_time += int(e.split(':')[1])
    tot_time.append(t_time)
    
    for e in h:
        eval('h' + str(e.split(':')[0]))[k] = int(e.split(':')[1])
        #eval('h' + str(e.split(':')[0]))[k] = 1
    
    for e in t:
        ss = e.split(':')[0]
        if ss in top_titles:
            ss = ss.replace(' ', '_')
            try:
                eval(ss)[k] = int(e.split(':')[1])
            except:
                eval(ss)[k] = 100
        
    avg_per_city.append((t_time*1.0)/len(c))
    avg_per_genre.append((t_time*1.0)/len(g))
    avg_per_title.append((t_time*1.0)/len(t))
    avg_per_day.append((t_time*1.0)/len(d))
    avg_per_hour.append((t_time*1.0)/len(h))
        
    k += 1
    
    if i%10000 == 0:
        print i


train_test['count_city'] = count_city
train_test['count_genre'] = count_genre
train_test['count_titles'] = count_titles
train_test['count_days'] = count_days
train_test['count_hours'] = count_hours

train_test['avg_per_city'] = avg_per_city
train_test['avg_per_genre'] = avg_per_genre
train_test['avg_per_title'] = avg_per_title
train_test['avg_per_day'] = avg_per_day
train_test['avg_per_hour'] = avg_per_hour

train_test['Koffee_With_Karan'] = Koffee_With_Karan
train_test['Yeh_Rishta_Kya_Kehlata_Hai'] = Yeh_Rishta_Kya_Kehlata_Hai
train_test['Ishqbaaaz'] = Ishqbaaaz
train_test['Ye_Hai_Mohabbatein'] = Ye_Hai_Mohabbatein
train_test['Chandra_Nandni'] = Chandra_Nandni
train_test['Naamkarann'] = Naamkarann
train_test['Pardes_Mein_Hai_Mera_Dil'] = Pardes_Mein_Hai_Mera_Dil
train_test['Tanhaiyan'] = Tanhaiyan
train_test['Dil_Boley_Oberoi'] = Dil_Boley_Oberoi
train_test['Jana_Na_Dil_Se_Door'] = Jana_Na_Dil_Se_Door
train_test['Saath_Nibhaana_Saathiya'] = Saath_Nibhaana_Saathiya
train_test['Mere_Angne_Mein'] = Mere_Angne_Mein
train_test['MS_Dhoni'] = MS_Dhoni
train_test['Nach_Baliye'] = Nach_Baliye
train_test['Ghulaam'] = Ghulaam
train_test['Suhani_Si_Ek_Ladki'] = Suhani_Si_Ek_Ladki
train_test['Dil_Hai_Hindustani'] = Dil_Hai_Hindustani
train_test['Savdhaan_India'] = Savdhaan_India
train_test['Koi_Laut_Ke_Aaya_Hai'] = Koi_Laut_Ke_Aaya_Hai
train_test['Sarabhai_Vs_Sarabhai'] = Sarabhai_Vs_Sarabhai

train_test['hyderabad'] = hyderabad
train_test['Unknown'] = Unknown
train_test['nagari'] = nagari
train_test['gandhinagar'] = gandhinagar
train_test['dubayy'] = dubayy
train_test['new_delhi'] = new_delhi
train_test['mumbai'] = mumbai
train_test['rawalpindi'] = rawalpindi
train_test['bengaluru'] = bengaluru
train_test['lahore'] = lahore
train_test['kolkata'] = kolkata
train_test['bhopal'] = bhopal
train_test['chandigarh'] = chandigarh
train_test['gurgaon'] = gurgaon
train_test['chennai'] = chennai
train_test['prabhadevi'] = prabhadevi
train_test['delhi'] = delhi
train_test['nagar'] = nagar
train_test['bangalore'] = bangalore
train_test['ahmedabad'] = ahmedabad
train_test['karachi'] = karachi
train_test['pune'] = pune
train_test['navi_mumbai'] = navi_mumbai
train_test['secunderabad'] = secunderabad
train_test['dhaka'] = dhaka

train_test['Travel'] = Travel
train_test['Kabaddi'] = Kabaddi
train_test['Crime'] = Crime
train_test['Romance'] = Romance
train_test['LiveTV'] = LiveTV
train_test['Hockey'] = Hockey
train_test['FormulaE'] = FormulaE
train_test['Comedy'] = Comedy
train_test['Teen'] = Teen
train_test['Cricket'] = Cricket
train_test['Mythology'] = Mythology
train_test['NA'] = NA
train_test['Horror'] = Horror
train_test['Football'] = Football
train_test['Awards'] = Awards
train_test['Science'] = Science
train_test['Tennis'] = Tennis
train_test['Thriller'] = Thriller
train_test['Boxing'] = Boxing
train_test['Wildlife'] = Wildlife
train_test['Kids'] = Kids
train_test['IndiaVsSa'] = IndiaVsSa
train_test['TalkShow'] = TalkShow
train_test['TableTennis'] = TableTennis
train_test['Volleyball'] = Volleyball
train_test['Drama'] = Drama
train_test['Action'] = Action
train_test['Athletics'] = Athletics
train_test['Reality'] = Reality
train_test['Documentary'] = Documentary
train_test['Swimming'] = Swimming
train_test['Formula1'] = Formula1
train_test['Family'] = Family
train_test['Badminton'] = Badminton
train_test['Sport'] = Sport

train_test['d1'] = d1
train_test['d2'] = d2
train_test['d3'] = d3
train_test['d4'] = d4
train_test['d5'] = d5
train_test['d6'] = d6
train_test['d7'] = d7

train_test['h0'] = h0
train_test['h1'] = h1
train_test['h2'] = h2
train_test['h3'] = h3
train_test['h4'] = h4
train_test['h5'] = h5
train_test['h6'] = h6
train_test['h7'] = h7
train_test['h8'] = h8
train_test['h9'] = h9
train_test['h10'] = h10
train_test['h11'] = h11
train_test['h12'] = h12
train_test['h13'] = h13
train_test['h14'] = h14
train_test['h15'] = h15
train_test['h16'] = h16
train_test['h17'] = h17
train_test['h18'] = h18
train_test['h19'] = h19
train_test['h20'] = h20
train_test['h21'] = h21
train_test['h22'] = h22
train_test['h23'] = h23

train_test['tot_time'] = tot_time

train_test['other_genres'] = train_test['Travel'] + train_test['Kabaddi'] + train_test['Crime'] + train_test['Hockey'] + train_test['FormulaE'] + train_test['Teen']+ train_test['Mythology'] + train_test['NA'] + train_test['Horror'] + train_test['Science'] + train_test['Tennis'] + train_test['Thriller'] + train_test['Boxing'] + train_test['Wildlife'] + train_test['IndiaVsSa'] + train_test['Volleyball'] + train_test['Athletics'] + train_test['Documentary'] + train_test['Swimming'] + train_test['Formula1'] + train_test['Badminton'] + train_test['Sport'] + train_test['TableTennis']

def process_id(raw):
    return int(raw.split('-')[1])

train_test['ID_num'] = train_test['ID'].apply(process_id)

train_test.to_csv('train_test.csv', index=False)

del h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23
del d1, d2, d3, d4, d5, d6, d7
del hyderabad, Unknown, nagari, gandhinagar, dubayy, new_delhi, mumbai, rawalpindi, bengaluru, lahore, kolkata, bhopal, chandigarh, gurgaon, chennai, prabhadevi, delhi, nagar, bangalore, ahmedabad, karachi, pune, navi_mumbai, secunderabad, dhaka


train_test['segment'] = train_test['segment'].astype('int')
train = train_test[:train.shape[0]]
test = train_test[train.shape[0]:]
predictors = ['count_city', 'count_days', 'count_genre', 'count_hours', 'count_titles', 'tot_time', 'Romance', 'LiveTV', 'Comedy', 'Cricket', 'Football', 'Awards', 'Kids', 'TalkShow', 'Drama', 'Action', 'Reality', 'Family', 'other_genres', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'hyderabad', 'Unknown', 'nagari', 'gandhinagar', 'dubayy', 'new_delhi', 'mumbai', 'rawalpindi', 'bengaluru', 'lahore', 'kolkata', 'bhopal', 'chandigarh', 'gurgaon', 'chennai', 'prabhadevi', 'delhi', 'nagar', 'bangalore', 'ahmedabad', 'karachi', 'pune', 'navi_mumbai', 'secunderabad', 'dhaka', 'Koffee_With_Karan', 'Yeh_Rishta_Kya_Kehlata_Hai', 'Ishqbaaaz', 'Ye_Hai_Mohabbatein', 'Chandra_Nandni', 'Naamkarann', 'Pardes_Mein_Hai_Mera_Dil', 'Tanhaiyan', 'Dil_Boley_Oberoi', 'Jana_Na_Dil_Se_Door', 'Saath_Nibhaana_Saathiya', 'Mere_Angne_Mein', 'MS_Dhoni', 'Nach_Baliye', 'Ghulaam', 'Suhani_Si_Ek_Ladki', 'Dil_Hai_Hindustani', 'Savdhaan_India', 'Koi_Laut_Ke_Aaya_Hai', 'Sarabhai_Vs_Sarabhai']

# making the model now
def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['segment'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
        
    alg.fit(dtrain[predictors], dtrain['segment'], eval_metric='auc')
    
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    print '\nModel Report:'
    print 'AUC (Train): ', metrics.roc_auc_score(dtrain['segment'], dtrain_predprob)
    
    return alg

print 'Training model_1...'
xgb1 = XGBClassifier(
    learning_rate = 0.1,
    n_estimators = 10000,
    max_depth = 4,
    gamma = 0,
    objective = 'binary:logistic',
    seed = 27
)
model_1 = modelfit(xgb1, train, predictors)

print 'Predictions in progress...'
submit = pd.DataFrame()
submit['ID'] = test['ID']
pred_1 = model_1.predict_proba(test[predictors])[:,1]
submit['segment'] = pred_1
submit.to_csv('submit.csv', index=False)
