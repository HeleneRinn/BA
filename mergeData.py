import pandas as pd

# Dateien in Dataframes
df1 = pd.read_csv('hiring_Data_Chi.csv')
df2 = pd.read_csv('hiring_Data_Ger.csv')
df3 = pd.read_csv('hiring_Data_Gha.csv')
df4 = pd.read_csv('hiring_Data_Jap.csv')
df5 = pd.read_csv('hiring_Data_Nor.csv')
df6 = pd.read_csv('hiring_Data_Pol.csv')
df7 = pd.read_csv('hiring_Data_Cze.csv')
df8 = pd.read_csv('hiring_Data_Tur.csv')
df9 = pd.read_csv('hiring_Data_USA.csv')
df10 = pd.read_csv('hiring_Data_Spa.csv')

# Zeilen mischen
df1 = df1.sample(frac=1, random_state=42)
df2 = df2.sample(frac=1, random_state=42)
df3 = df3.sample(frac=1, random_state=42)
df4 = df4.sample(frac=1, random_state=42)
df5 = df5.sample(frac=1, random_state=42)

# Dataframes zusammenfügen
merged_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)

#Indexzeile erstellen
merged_df['Index'] = range(1, len(merged_df)+1)

# Ausgabe
print(merged_df.to_string())

merged_df.to_csv('hiring_Data.csv', index=False)

