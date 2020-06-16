import pandas as pd
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    #READ INPUT FILE
    dataframe = pd.read_csv("credit_card_data.csv")

    #DROP COLUMN ID
    dataframe = dataframe.drop("CUST_ID", 1)

    #GET THE NUMBER OF MISSING VALUES BY COLUMN
    print("Before imputations:")
    print(len(dataframe) - dataframe.count())

    #MINIMUM_PAYMENTS HAS MORE THAN 300 MISSING VALUES
    #CREDIT_LIMIT HAS ALSO MISSING VALUES
    #IMPUTE THEM USING MULTIPLE IMPUTATION (MICE)
    mice = IterativeImputer()
    dataframe.iloc[:, :] = mice.fit_transform(dataframe)

    print("\nAfter imputations:")
    print(len(dataframe) - dataframe.count()) #all columns should now have 0 misssing values

    #SCALE ALL VARIABLES
    standard_scaler = StandardScaler()
    dataframe = standard_scaler.fit_transform(dataframe)

    return dataframe

if __name__ == '__main__':
    load_and_preprocess()