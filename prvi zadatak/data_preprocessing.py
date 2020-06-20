import pandas as pd
#from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def outlier_treatment(x):
    return x.clip(lower=x.quantile(0.05), upper=x.quantile(0.95))

def load_and_preprocess():
    #READ INPUT FILE
    dataframe = pd.read_csv("credit_card_data.csv")

    #DROP COLUMN ID
    dataframe = dataframe.drop("CUST_ID", 1)

    #GET THE NUMBER OF MISSING VALUES BY COLUMN
    #print("Before imputations:")
    #print(len(dataframe) - dataframe.count())

    #MINIMUM_PAYMENTS HAS MORE THAN 300 MISSING VALUES
    #CREDIT_LIMIT HAS ALSO MISSING VALUES
    #IMPUTE THEM USING MULTIPLE IMPUTATION (MICE)
    #mice = IterativeImputer()
    #dataframe.iloc[:, :] = mice.fit_transform(dataframe)

    #impute with median, much simpler
    dataframe['CREDIT_LIMIT'].fillna(dataframe['CREDIT_LIMIT'].median(), inplace=True)
    dataframe['MINIMUM_PAYMENTS'].fillna(dataframe['MINIMUM_PAYMENTS'].median(), inplace=True)

    #print("\nAfter imputations:")
    #print(len(dataframe) - dataframe.count()) #all columns should now have 0 misssing values

    # ADDING SOME DERIVED FEATURES - KPI

    # 1. BALANCE TO CREDIT LIMIT RATIO - SHOULD BE LOW IF CREDIT CARD USER IS RESPONSIBLE
    dataframe["BALANCE_TO_CREDIT_LIMIT_RATIO"] = dataframe["BALANCE"] / dataframe["CREDIT_LIMIT"]

    # 2. PAYMENTS TO MINIMUM PAYMENTS RATIO - SHOULD BE HIGH IF CREDIT CARD USER IS RESPONSIBLE
    dataframe["PAYMENT_TO_MIN_PAYMENT_RATIO"] = dataframe["PAYMENTS"] / dataframe["MINIMUM_PAYMENTS"]

    # 3. MONTHLY AVERAGE PURCHASES AND CASH ADVANCE - IMPORTANT SINCE TENURE VARIES SIGNIFICANTLY AMONG CREDIT CARD HOLDERS
    dataframe["MONTHLY_AVERAGE_PURCHASES"] = dataframe["PURCHASES"] / dataframe["TENURE"]
    dataframe["MONTHLY_AVERAGE_CASH_ADVANCE"] = dataframe["CASH_ADVANCE"] / dataframe["TENURE"]

    #dataframe.plot(kind='box')
    #plt.show()
    # perform quantile clipping in order to treat outliers
    dataframe = dataframe.applymap(lambda x: np.log(x + 1))
    #dataframe.plot(kind='box')
    #plt.show()

    #SCALE ALL VARIABLES
    standard_scaler = StandardScaler()
    scaled = standard_scaler.fit_transform(dataframe.values)
    dataframe = pd.DataFrame(scaled, index=dataframe.index, columns=dataframe.columns)

    return dataframe

if __name__ == '__main__':
    load_and_preprocess()