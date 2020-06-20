from data_preprocessing import load_and_preprocess
import seaborn as sb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
def main():
    chosen_columns = ["BALANCE_TO_CREDIT_LIMIT_RATIO", "PAYMENT_TO_MIN_PAYMENT_RATIO", "MONTHLY_AVERAGE_PURCHASES", "MONTHLY_AVERAGE_CASH_ADVANCE"]
    X = load_and_preprocess()
    X_chosen = pd.DataFrame(X[chosen_columns])

    kmeans = KMeans(n_clusters=5)
    label = kmeans.fit_predict(X_chosen)
    # create a 'cluster' column
    X_chosen['cluster'] = label
    chosen_columns.append('cluster')
    # make a Seaborn pairplot
    sb.countplot(data=X_chosen, hue='cluster', x='cluster')
    sb.pairplot(X_chosen, hue='cluster', vars=chosen_columns)
    plt.show()
if __name__ == '__main__':
    main()