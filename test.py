from classes.preparation import Preparation
import pandas as pd

pipeline = Preparation()
df = pd.read_csv("data_preprocessed.csv")
pipeline.transform(df)