import pandas as pd

df = pd.read_csv("data/Malaria-Data.csv")

print(df.groupby("severe_maleria")[[
    "Convulsion",
    "prostraction",
    "hypoglycemia",
    "diarrhea",
    "headace"
]].mean())