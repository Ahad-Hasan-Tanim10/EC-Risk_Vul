# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:54:10 2023

@author: ATANIM
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 

#%%
HPI = pd.read_excel(r"\\coeciv-nas05.cec.sc.edu\covid19\Natural Hazard Vulnerability\data\House Price INDEX\HPI_County_yr.xlsx", 'Sheet1', index_col = 0, header= 0)
Hpi_T = HPI.transpose()
#Hpi_T['slope'] = None

# Create an instance of the LinearRegression model
model = LinearRegression()
#%%
X = Hpi_T.index.values
Y = Hpi_T[1001].dropna()

X = Y.index.values.reshape(-1, 1)
y = Y.values.reshape(-1, 1)


model.fit(X, y)
slope = model.coef_[0][0]
#%%
sl_L = []

for name, values in Hpi_T.iteritems():
    # Drop rows with NaN values for the current column
    Yx = values.dropna()
    
    # Extract the features (X) and target (y)
    X = np.array(Yx.index).reshape(-1, 1)
    y = np.array(Yx.values).reshape(-1, 1)
    
    # Fit the model on the features and target
    model.fit(X, y)
    
    # Get the regression slope (coefficient)
    slope = model.coef_[0][0]
    
    # Append the slope to the list
    sl_L.append(slope)
    
#%%
FIPS = HPI.index.values
comb = list(zip(FIPS, sl_L))
HPI_c = pd.DataFrame(comb, columns=['FIPS', 'HPI_rate'])
HPI_c.to_csv(r'\\coeciv-nas05.cec.sc.edu\covid19\Natural Hazard Vulnerability\data\House Price INDEX\HPI_c.csv', index=False)
