import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

colors = {
    'lsa': '#333333',  # Black
    'divider': '#FF8C00',  # Orange
    'bucket': '#800080',  # Purple
    'direct': '#0000FF'  # Blue
}

cost_path = '../csv/evaluation_costs.csv'
acc_path = '../csv/evaluation_accuracy.csv'
timing_path = '../csv/timing_evaluation_detailed.csv'

cost_pd = pd.read_csv(cost_path)
acc_pd = pd.read_csv(acc_path)
timing_pd = pd.read_csv(timing_path)

def show_dataframes():
    print("Cost DataFrame:")
    print(cost_pd.head())
    print("\nAccuracy DataFrame:")
    print(acc_pd.head())
    print("\nTiming DataFrame:")
    print(timing_pd.head())


show_dataframes()

