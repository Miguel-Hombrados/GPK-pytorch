#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:50:38 2020

@author: Miguel  A Hombrados
Descrption: Script for reading the ISO NE dataset for load profiling in the context
of the paper of NMF Correlated. It takes time series of real time demand, dew point,
and temperature of a particular load zone selected by "location":
    0:  ME
    1:  NH
    2:  VT
    3:  CT
    4:  RI
    5:  SEMASS
    6:  WCMASS
    7:  NEMASSBOST
        

Output: Data_test and Data_train, both of them data structures containing:
        Date, Day of the year, 24 values of hourly Real time,24 values of hourly Temperature,
        24 values of hourly Dew point and the Weekday. The split into train and test of 
        the whole data set is defined by a date specified by the variables "day", "month" and  "year/"

"""

import pandas as pd
import datetime
import scipy
import scipy.io
import numpy as np
import pickle
from pathlib import Path

LOCATIONS = ['ME','NH','VT','CT','RI','SEMASS','WCMASS','NEMASSBOST']
project_path = Path("/Users/apple/Desktop/PASAR")
#==================================================================
# SELEECT DATE THAT SPLITS DATA SET INTO TRAIN AND TEST
#==================================================================
#==================================================================
start_day_train_val = 1
start_month_train_val = 1
start_year_train_val= 2011
end_day_train_val = 31
end_month_train_val = 12
end_year_train_val = 2017



start_day_test = 1
start_month_test = 1
start_year_test = 2018
end_day_test = 31
end_month_test = 12
end_year_test = 2018
#==================================================================

data_folder = Path("/Users/apple/Desktop/PASAR/ISO_NE_Dataset_Final/Nestor")

filename = "iso_ne.pickle"
file_to_open = data_folder / filename
pickle_in=open(file_to_open,'rb')
iso_ne=pickle.load(pickle_in)



for location in range(0,8):

    location_name = LOCATIONS[location]
    
    data2011=iso_ne[location][0]
    data2012=iso_ne[location][1]
    data2013=iso_ne[location][2]
    data2014=iso_ne[location][3]
    data2015=iso_ne[location][4]
    data2016=iso_ne[location][5]
    data2017=iso_ne[location][6]
    data2018=iso_ne[location][7]
    
    
    Y2011=data2011[['Date','Hour','DEMAND','DryBulb','DewPnt']]
    Y2012=data2012[['Date','Hour','DEMAND','DryBulb','DewPnt']]
    Y2013=data2013[['Date','Hour','DEMAND','DryBulb','DewPnt']]
    Y2014=data2014[['Date','Hour','DEMAND','DryBulb','DewPnt']]
    Y2015=data2015[['Date','Hour','DEMAND','DryBulb','DewPnt']]
    Y2016=data2016[['Date','Hr_End','RT_Demand','Dry_Bulb','Dew_Point']]
    Y2017=data2017[['Date','Hr_End','RT_Demand','Dry_Bulb','Dew_Point']]
    Y2018=data2018[['Date','Hr_End','RT_Demand','Dry_Bulb','Dew_Point']]
    
    Aux2011 = pd.to_datetime(Y2011['Date']).dt.strftime('%d-%b-%Y')
    Dates2011 = pd.Series(list(Aux2011[0::24]))
    DoWeek2011 = pd.to_datetime(Dates2011).dt.day_name()
    Load2011 = pd.Series(list(Y2011['DEMAND'].values.reshape(-1,24)))
    Temperature2011 = pd.Series(list(Y2011['DryBulb'].values.reshape(-1,24)))
    DewPoint2011 = pd.Series(list(Y2011['DewPnt'].values.reshape(-1,24)))
    del Y2011
    frame2011 = { 'Date': Dates2011, 'Weekday': DoWeek2011} 
    frame2011['Load'] = list(Load2011)
    frame2011['Temperature'] = list(Temperature2011)
    frame2011['DewPoint'] = list(DewPoint2011)
    Y2011 = pd.DataFrame(frame2011) 
    
    Aux2012 = pd.to_datetime(Y2012['Date']).dt.strftime('%d-%b-%Y')
    Dates2012 = pd.Series(list(Aux2012[0::24]))
    DoWeek2012 = pd.to_datetime(Dates2012).dt.day_name()
    Load2012 = pd.Series(list(Y2012['DEMAND'].values.reshape(-1,24)))
    Temperature2012 = pd.Series(list(Y2012['DryBulb'].values.reshape(-1,24)))
    DewPoint2012 = pd.Series(list(Y2012['DewPnt'].values.reshape(-1,24)))
    del Y2012
    frame2012 = { 'Date': Dates2012, 'Weekday': DoWeek2012} 
    frame2012['Load'] = list(Load2012)
    frame2012['Temperature'] = list(Temperature2012)
    frame2012['DewPoint'] = list(DewPoint2012)
    Y2012 = pd.DataFrame(frame2012) 
    
    Aux2013 = pd.to_datetime(Y2013['Date']).dt.strftime('%d-%b-%Y')
    Dates2013 = pd.Series(list(Aux2013[0::24]))
    DoWeek2013 = pd.to_datetime(Dates2013).dt.day_name()
    Load2013 = pd.Series(list(Y2013['DEMAND'].values.reshape(-1,24)))
    Temperature2013 = pd.Series(list(Y2013['DryBulb'].values.reshape(-1,24)))
    DewPoint2013 = pd.Series(list(Y2013['DewPnt'].values.reshape(-1,24)))
    del Y2013
    frame2013 = { 'Date': Dates2013, 'Weekday': DoWeek2013} 
    frame2013['Load'] = list(Load2013)
    frame2013['Temperature'] = list(Temperature2013)
    frame2013['DewPoint'] = list(DewPoint2013)
    Y2013 = pd.DataFrame(frame2013) 
    
    Aux2014 = pd.to_datetime(Y2014['Date']).dt.strftime('%d-%b-%Y')
    Dates2014 = pd.Series(list(Aux2014[0::24]))
    DoWeek2014 = pd.to_datetime(Dates2014).dt.day_name()
    Load2014 = pd.Series(list(Y2014['DEMAND'].values.reshape(-1,24)))
    Temperature2014 = pd.Series(list(Y2014['DryBulb'].values.reshape(-1,24)))
    DewPoint2014 = pd.Series(list(Y2014['DewPnt'].values.reshape(-1,24)))
    del Y2014
    frame2014 = { 'Date': Dates2014, 'Weekday': DoWeek2014} 
    frame2014['Load'] = list(Load2014)
    frame2014['Temperature'] = list(Temperature2014)
    frame2014['DewPoint'] = list(DewPoint2014)
    Y2014 = pd.DataFrame(frame2014) 
    
    Aux2015 = pd.to_datetime(Y2015['Date']).dt.strftime('%d-%b-%Y')
    Dates2015 = pd.Series(list(Aux2015[0::24]))
    DoWeek2015 = pd.to_datetime(Dates2015).dt.day_name()
    Load2015 = pd.Series(list(Y2015['DEMAND'].values.reshape(-1,24)))
    Temperature2015 = pd.Series(list(Y2015['DryBulb'].values.reshape(-1,24)))
    DewPoint2015 = pd.Series(list(Y2015['DewPnt'].values.reshape(-1,24)))
    del Y2015
    frame2015 = { 'Date': Dates2015, 'Weekday': DoWeek2015} 
    frame2015['Load'] = list(Load2015)
    frame2015['Temperature'] = list(Temperature2015)
    frame2015['DewPoint'] = list(DewPoint2015)
    Y2015 = pd.DataFrame(frame2015) 
    
    Aux2016 = pd.to_datetime(Y2016['Date']).dt.strftime('%d-%b-%Y')
    Dates2016 = pd.Series(list(Aux2016[0::24]))
    DoWeek2016 = pd.to_datetime(Dates2016).dt.day_name()
    Load2016 = pd.Series(list(Y2016['RT_Demand'].values.reshape(-1,24)))
    Temperature2016 = pd.Series(list(Y2016['Dry_Bulb'].values.reshape(-1,24)))
    DewPoint2016 = pd.Series(list(Y2016['Dew_Point'].values.reshape(-1,24)))
    del Y2016
    frame2016 = { 'Date': Dates2016, 'Weekday': DoWeek2016} 
    frame2016['Load'] = list(Load2016)
    frame2016['Temperature'] = list(Temperature2016)
    frame2016['DewPoint'] = list(DewPoint2016)
    Y2016 = pd.DataFrame(frame2016) 
    
    Aux2017 = pd.to_datetime(Y2017['Date']).dt.strftime('%d-%b-%Y')
    Dates2017 = pd.Series(list(Aux2017[0::24]))
    DoWeek2017 = pd.to_datetime(Dates2017).dt.day_name()
    Load2017 = pd.Series(list(Y2017['RT_Demand'].values.reshape(-1,24)))
    Temperature2017 = pd.Series(list(Y2017['Dry_Bulb'].values.reshape(-1,24)))
    DewPoint2017 = pd.Series(list(Y2017['Dew_Point'].values.reshape(-1,24)))
    del Y2017
    frame2017 = { 'Date': Dates2017, 'Weekday': DoWeek2017} 
    frame2017['Load'] = list(Load2017)
    frame2017['Temperature'] = list(Temperature2017)
    frame2017['DewPoint'] = list(DewPoint2017)
    Y2017 = pd.DataFrame(frame2017) 
    
    Aux2018 = pd.to_datetime(Y2018['Date']).dt.strftime('%d-%b-%Y')
    Dates2018 = pd.Series(list(Aux2018[0::24]))
    DoWeek2018 = pd.to_datetime(Dates2018).dt.day_name()
    Load2018 = pd.Series(list(Y2018['RT_Demand'].values.reshape(-1,24)))
    Temperature2018 = pd.Series(list(Y2018['Dry_Bulb'].values.reshape(-1,24)))
    DewPoint2018 = pd.Series(list(Y2018['Dew_Point'].values.reshape(-1,24)))
    del Y2018
    frame2018 = { 'Date': Dates2018, 'Weekday': DoWeek2018} 
    frame2018['Load'] = list(Load2018)
    frame2018['Temperature'] = list(Temperature2018)
    frame2018['DewPoint'] = list(DewPoint2018)
    Y2018 = pd.DataFrame(frame2018) 
    
    
    Yeardayindex2011 = np.array(range(1,np.size(Y2011,0)+1)) 
    Yeardayindex2012 = np.array(range(1,np.size(Y2012,0)+1))
    Yeardayindex2013 = np.array(range(1,np.size(Y2013,0)+1))
    Yeardayindex2014 = np.array(range(1,np.size(Y2014,0)+1))
    Yeardayindex2015 = np.array(range(1,np.size(Y2015,0)+1))
    Yeardayindex2016 = np.array(range(1,np.size(Y2016,0)+1))
    Yeardayindex2017 = np.array(range(1,np.size(Y2017,0)+1))
    Yeardayindex2018 = np.array(range(1,np.size(Y2018,0)+1))
    
    DaysIndex = np.concatenate((Yeardayindex2011,Yeardayindex2012,Yeardayindex2013,Yeardayindex2014,Yeardayindex2015,Yeardayindex2016,Yeardayindex2017,Yeardayindex2018))
    GeneralIndex = np.array(range(1,len(DaysIndex)+1))
    
    DATA = pd.concat([Y2011,Y2012,Y2013,Y2014,Y2015,Y2016,Y2017,Y2018], ignore_index=True)
    DATA['DayOfYear'] = DaysIndex 
    
    LOAD_DATA = DATA.apply(tuple).to_dict()
    
    # Split into train and test================
    
    DATA2 = pd.DataFrame.copy(DATA)
    DATA2['Date'] = pd.to_datetime(DATA2['Date']).dt.date
    

    DATA_Test = DATA2[(DATA2['Date']<=datetime.date(end_year_test,end_month_test,end_day_test)) &  (DATA2['Date']>=datetime.date(start_year_test,start_month_test,start_day_test))]
  
    DATA_Train_Val = DATA2[(DATA2['Date']<=datetime.date(end_year_train_val,end_month_train_val,end_day_train_val))  & (DATA2['Date']>=datetime.date(start_year_train_val,start_month_train_val,start_day_train_val))]
    
    Dates_Test = pd.to_datetime(DATA_Test['Date']).dt.strftime('%d-%b-%Y')
    DATA_Test_aux = DATA_Test.drop(['Date'],axis=1)
    DATA_Test = {** DATA_Test_aux.to_dict("list"),**{'Date':list(Dates_Test)}}

    Dates_Train_Val = pd.to_datetime(DATA_Train_Val['Date']).dt.strftime('%d-%b-%Y')
    DATA_Train_Val_aux = DATA_Train_Val.drop(['Date'],axis=1)
    DATA_Train_Val = {** DATA_Train_Val_aux.to_dict("list"),**{'Date':list(Dates_Train_Val)}}

    
    data_test_name = "Data/Full/DATA_Test_11_18"+str(location_name)+".mat"
    data_train_val_name = "Data/Full/DATA_Train_Val_11_18"+str(location_name)+".mat"
    
    scipy.io.savemat(project_path/data_test_name, DATA_Test)
    scipy.io.savemat(project_path/ data_train_val_name, DATA_Train_Val)




