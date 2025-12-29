import pandas as pd

df = pd.read_csv('Raw_Data.csv')
df.drop('StudentID', axis=1)

df["ExtracurricularActivities"] = df["ExtracurricularActivities"].astype('category')
df["ExtracurricularActivities"] = df["ExtracurricularActivities"].cat.codes
df["PlacementTraining"] = df["PlacementTraining"].astype('category')
df["PlacementTraining"] = df["PlacementTraining"].cat.codes
df["PlacementStatus"] = df["PlacementStatus"].astype('category')
df["PlacementStatus"] = df["PlacementStatus"].cat.codes

df.to_csv('Dataset.csv', index=False)