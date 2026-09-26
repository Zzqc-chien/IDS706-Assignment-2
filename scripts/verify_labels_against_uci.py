import pandas as pd

cols = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
orig = pd.read_csv(url, names=cols, na_values="?")
ours = pd.read_csv("heart.csv", encoding="utf-8-sig")

print("ORIGINAL disease present (num>0):")
print((orig["num"] > 0).value_counts())
print("\nOURS target:")
print(ours["target"].value_counts())
print("\nORIGINAL cp (1=typical ... 4=asymptomatic):")
print(orig["cp"].value_counts().sort_index())
print("\nOURS cp:")
print(ours["cp"].value_counts().sort_index())
print("\nORIGINAL missing values:")
print(orig[["ca", "thal"]].isna().sum())
