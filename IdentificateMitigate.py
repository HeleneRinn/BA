import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2_contingency

plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

# Laden des Datensets
df = pd.read_csv("hiring_Data.csv")

# Hiring von Boolean zu Int
df['Hired'] = df['Hired'].replace(True, 1)
df['Hired'] = df['Hired'].replace(False, 0)

# Gender von String zu Int
df['Gender'] = df['Gender'].replace('Male', 0)
df['Gender'] = df['Gender'].replace('Female', 1)

# One-Hot Encoding für Job Description und Education
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(df[['Education', 'Job Description']]).toarray(), columns=enc.get_feature_names_out(['Education', 'Job Description']))
df = df.drop(['Education', 'Job Description'], axis=1)
df = df.join(enc_df)

# MultiLabelBinarizer für Skills und Languages
#mlb = MultiLabelBinarizer()
#df['Skills'] = df['Skills'].apply(eval)  # konvertieren des String zurück in eine Liste
#df['Languages'] = df['Languages'].apply(eval)
#skills_encoded = pd.DataFrame(mlb.fit_transform(df['Skills']), columns=mlb.classes_, index=df.index)
#skills = skills_encoded.columns
#languages_encoded = pd.DataFrame(mlb.fit_transform(df['Languages']), columns=mlb.classes_, index=df.index)
#df = df.drop(['Skills', 'Languages'], axis=1)
#df = df.join(skills_encoded).join(languages_encoded)

# Namen entfernen
df = df.drop(columns=['First Name', 'Last Name', 'Skills', 'Languages'])
df.to_csv('hiring_Data_encoded.csv', index=False)

# XGBoost
df = pd.read_csv('hiring_Data_encoded.csv')
# Trennung von Features und Target
X = df.drop(['Hired', 'Index'], axis=1)
y = df['Hired']

# Aufteilung in Trainings- und Testdatensätze
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=X['Gender'])

# Mean Difference
df_diff = pd.concat([X_train, y_train], axis = 1)
mean_male = df_diff[df_diff['Gender'] == 0]['Hired'].mean()
mean_female = df_diff[df_diff['Gender'] == 1]['Hired'].mean()
mean_diff = mean_female - mean_male
print('Mean difference:', mean_diff)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(f'train_size: {X_train.shape[0]}')
# XGBoost Modell initialisieren (vor Bias identifikation und Mitigation)
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric=['auc']) #binary:logistik, weil binäre Klassifikation, auc, um die Modelleistung zu bewerten
# Modell trainieren
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
# Vorhersagen auf Testdaten treffen
y_pred = xgb_model.predict(X_test)
# Genauigkeit berechnen
accuracy = accuracy_score(y_test, y_pred)
print("Genauigkeit vorher:", accuracy)
print(f'Recall: {recall_score(y_test, y_pred)}')
confusionbefore = (confusion_matrix(y_test, y_pred))
print(confusionbefore)
# Ergebnis als Balkendiagramm ausgeben
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [confusionbefore[0, 0], confusionbefore[0, 1], confusionbefore[1, 0], confusionbefore[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.savefig('Keine_methodik_male_female.png', dpi=my_dpi * 2)
plt.show()

# evaluation male
df_test = pd.concat([X_test, y_test], axis = 1)
df_test_male = df_test[df_test['Gender']==0]
X_test_male = df_test_male.drop(['Hired'], axis=1)
y_test_male = df_test_male['Hired']
y_pred_male = xgb_model.predict(X_test_male)
# Genauigkeit berechnen
accuracy_male = accuracy_score(y_test_male, y_pred_male)
print("Genauigkeit vorher male:", accuracy_male)
print(f'Recall: {recall_score(y_test_male, y_pred_male)}')
print(confusion_matrix(y_test_male, y_pred_male))
# Ergebnis als Balkendiagramm ausgeben male
fig, ax = plt.subplots()
#plt.figure(figsize=(20, 16), dpi=100)
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [confusionbefore_male[0, 0], confusionbefore_male[0, 1], confusionbefore_male[1, 0], confusionbefore_male[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix male')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.savefig('Keine_methodik_male.png', dpi=my_dpi * 2)
plt.show()

# evaluation female
df_test_female = df_test[df_test['Gender'] == 1]
X_test_female = df_test_female.drop(['Hired'], axis=1)
y_test_female = df_test_female['Hired']
y_pred_female = xgb_model.predict(X_test_female)
# Genauigkeit berechnen
accuracy_female = accuracy_score(y_test_female, y_pred_female)
print("Genauigkeit vorher female:", accuracy_female)
print(f'Recall: {recall_score(y_test_female, y_pred_female)}')
confusionbefore_female = (confusion_matrix(y_test_female, y_pred_female))
print(confusionbefore_female)
# Ergebnis als Balkendiagramm ausgeben male
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [confusionbefore_female[0, 0], confusionbefore_female[0, 1], confusionbefore_female[1, 0], confusionbefore_female[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix female')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.savefig('Keine_methodik_female.png', dpi=my_dpi * 2)
plt.show()# Feature importance einzelne Features

xgb.plot_importance(xgb_model)

# Genderswap zur Identifikation
#X_test_male_to_female = X_test_male.copy()
#X_test_male_to_female['Gender'] = X_test_male_to_female['Gender'].replace({0:1})
#y_test_male_to_female = y_test_male.copy()
#y_pred_male_to_female = xgb_model.predict(X_test_male_to_female)
# Genauigkeit berechnen male to female
#accuracy_male_to_female = accuracy_score(y_test_male_to_female, y_pred_male_to_female)
#print("Genauigkeit swap male to female:", accuracy_male_to_female)
#print(confusion_matrix(y_test_male_to_female, y_pred_male_to_female))
#print(f'Recall: {recall_score(y_test_male_to_female, y_pred_male_to_female)}')

# Genderswap zur Identifikation
#X_test_female_to_male = X_test_female.copy()
#X_test_female_to_male['Gender'] = X_test_female_to_male['Gender'].replace({1:0})
#y_test_female_to_male = y_test_female.copy()
#y_pred_female_to_male = xgb_model.predict(X_test_female_to_male)
# Genauigkeit berechnen female to male
#accuracy_female_to_male = accuracy_score(y_test_female_to_male, y_pred_female_to_male)
#print("Genauigkeit swap female to male:", accuracy_female_to_male)
#print(confusion_matrix(y_test_female_to_male, y_pred_female_to_male))
#print(f'Recall: {recall_score(y_test_female_to_male, y_pred_female_to_male)}')


# Evaluation nach Trennung der Geschlechter Confusion Matrix
#y_men_test = [y_test['Hired'] == 1]
#y_female_test = [y_test['Hired'] == 1]




# Gender Variable löschen
df_withoutgender = df.copy()
df_withoutgender.drop('Gender', axis=1)
# Gesamten Datensatz in Trainings- und Testdaten aufteilen
# Gesamten Datensatz in Trainings- und Testdaten aufteilen
X = df_withoutgender.drop('Hired', axis=1)
y = df_withoutgender['Hired']

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X, y, test_size=0.2, random_state=42)

# Mean Difference
df_diff_no_g = pd.concat([X_train_g, y_train_g], axis = 1)
mean_male = df_diff_no_g[df_diff_no_g['Gender'] == 0]['Hired'].mean()
mean_female = df_diff_no_g[df_diff_no_g['Gender'] == 1]['Hired'].mean()
mean_diff = mean_female - mean_male
print('Mean difference:', mean_diff)

X_train_no_g = X_train_g.drop('Gender', axis=1)
X_test_no_g = X_test_g.drop('Gender', axis=1)
X_train_no_g, X_val_no_g, y_train_no_g, y_val_no_g = train_test_split(X_train_no_g, y_train_g, test_size=0.2, random_state=42)

# XGBoost-Modelle trainieren
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric=['auc'])
xgb_model.fit(X_train_no_g, y_train_no_g, eval_set=[(X_val_no_g, y_val_no_g)])

# Vorhersagen für Testdatensatz durchführen
y_pred_g = xgb_model.predict(X_test_no_g)

# Genauigkeit, Recall und Confusion Matrix für das Modell berechnen
g_accuracy = accuracy_score(y_test_g, y_pred_g)
g_recall = recall_score(y_test_g, y_pred_g)
g_confusion = confusion_matrix(y_test_g, y_pred_g)
# Ergebnisse ausgeben
print("Genauigkeit Datensatz ohne gender:", g_accuracy)
print("Recall Datensatz ohne gender:", g_recall)
print("Confusion Matrix Datensatz ohne gender:", g_confusion)
# Ergebnis als Balkendiagramm ausgeben
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [g_confusion[0, 0], g_confusion[0, 1], g_confusion[1, 0], g_confusion[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
#ax.set_title('Confusion Matrix')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
#plt.title('Klassifikationsdiagramm nach')
plt.savefig('remove_gender_methodik_all.png', dpi=my_dpi * 2)
plt.show()

# evaluation male
X_test_g_male = X_test_g[X_test_g['Gender']==0]
df_test_male = pd.concat([X_test_g_male, y_test_g], axis = 1)
X_test_no_g_male = df_test_male.drop(['Gender', 'Hired'], axis=1)
y_test_no_g_male = df_test_male['Hired']
y_pred_no_g_male = xgb_model.predict(X_test_no_g_male)
# Genauigkeit, Recall und Confusion Matrix für das Modell berechnen
no_g_accuracy_male = accuracy_score(y_test_no_g_male, y_pred_no_g_male)
no_g_recall_male = recall_score(y_test_no_g_male, y_pred_no_g_male)
no_g_confusion_male = confusion_matrix(y_test_no_g_male, y_pred_no_g_male)
# Ergebnisse ausgeben
print("Genauigkeit Datensatz ohne gender male:", no_g_accuracy_male)
print("Recall Datensatz ohne gender male:", no_g_recall_male)
print("Confusion Matrix Datensatz ohne gender male:", no_g_confusion_male)
# Ergebnis als Balkendiagramm ausgeben
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [no_g_confusion_male[0, 0], no_g_confusion_male[0, 1], no_g_confusion_male[1, 0], no_g_confusion_male[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
#ax.set_title('Confusion Matrix')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
#plt.title('Confusion Matrix after removing Gender for male')
plt.savefig('remove_gender_methodik_male.png', dpi=my_dpi * 2)
plt.show()
#df_withoutgender.to_csv('without_gender.csv', index=False)

# evaluation female
X_test_g_female = X_test_g[X_test_g['Gender']==1]
df_test_female = pd.concat([X_test_g_female, y_test_g], axis = 1)
X_test_no_g_female = df_test_female.drop(['Gender', 'Hired'], axis=1)
y_test_no_g_female = df_test_female['Hired']
y_pred_no_g_female = xgb_model.predict(X_test_no_g_female)
# Genauigkeit, Recall und Confusion Matrix für das Modell berechnen
no_g_accuracy_female = accuracy_score(y_test_no_g_female, y_pred_no_g_female)
no_g_recall_female = recall_score(y_test_no_g_female, y_pred_no_g_female)
no_g_confusion_female = confusion_matrix(y_test_no_g_female, y_pred_no_g_female)
# Ergebnisse ausgeben
print("Genauigkeit Datensatz ohne gender female:", no_g_accuracy_female)
print("Recall Datensatz ohne gender female:", no_g_recall_female)
print("Confusion Matrix Datensatz ohne gender female:", no_g_confusion_female)
# Ergebnis als Balkendiagramm ausgeben
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [no_g_confusion_female[0, 0], no_g_confusion_female[0, 1], no_g_confusion_female[1, 0], no_g_confusion_female[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.title('Confusion Matrix after removing Gender for female')
plt.savefig('remove_gender_methodik_female.png', dpi=my_dpi * 2)
plt.show()
#df_withoutgender.to_csv('without_gender.csv', index=False)

xgb.plot_importance(xgb_model)





# Anwendung Gender Swapping auf die Traningsdaten zur Bias mitigation
# Testdatensatz duplizieren und tauschen
# Geschlecht umkehren
# Geschlecht umkehren
df_swapped = df.copy()
X = df.drop('Hired', axis=1)
y = df['Hired']
X_train, X_test_s, y_train, y_test_s = train_test_split(X, y, test_size=0.2, random_state=42)
train = pd.concat([X_train, y_train], axis = 1)
train_s = pd.concat([X_train, y_train], axis = 1)
train_s['Gender'] = train_s['Gender'].replace({0: 1, 1: 0})
X_train_s = pd.concat([train.drop('Hired', axis=1), train_s.drop('Hired', axis=1)])
y_train_s = pd.concat([train['Hired'], train_s['Hired']])

# Mean Difference
df_diff_s = pd.concat([X_train_s, y_train_s], axis = 1)
mean_male = df_diff_s[df_diff_s['Gender'] == 0]['Hired'].mean()
mean_female = df_diff_s[df_diff_s['Gender'] == 1]['Hired'].mean()
mean_diff = mean_female - mean_male
print('Mean difference:', mean_diff)

X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_s, y_train_s, test_size=0.2, random_state=42)
# XGBoost-Modelle trainieren
xgb_model_s = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric=['auc'])
xgb_model_s.fit(X_train_s, y_train_s, eval_set=[(X_val_s, y_val_s)])
# Vorhersagen für Testdatensatz durchführen
y_pred_s = xgb_model_s.predict(X_test_s)

# Genauigkeit, Recall und Confusion Matrix für das Modell berechnen
swapped_accuracy = accuracy_score(y_test_s, y_pred_s)
swapped_recall = recall_score(y_test_s, y_pred_s)
swapped_confusion = confusion_matrix(y_test_s, y_pred_s)
# Ergebnisse ausgeben
print("Genauigkeit geswapter Datensatz:", swapped_accuracy)
print("Recall geswapter Datensatz:", swapped_recall)
print("Confusion Matrix geswapter Datensatz:", swapped_confusion)
# Ergebnis als Balkendiagramm ausgeben
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [swapped_confusion[0, 0], swapped_confusion[0, 1], swapped_confusion[1, 0], swapped_confusion[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.title('Confusion Matrix after Gender Swapping')
plt.savefig('swapping_methodik_all.png', dpi=my_dpi * 2)
plt.show()

# evaluation male
df_test_s = pd.concat([X_test_s, y_test_s], axis = 1)
df_test_s_male = df_test_s[df_test_s['Gender']==0]
X_test_s_male = df_test_s_male.drop(['Hired'], axis=1)
y_test_s_male = df_test_s_male['Hired']
y_pred_s_male = xgb_model_s.predict(X_test_s_male)
# Genauigkeit berechnen
accuracy_s_male = accuracy_score(y_test_s_male, y_pred_s_male)
print("Genauigkeit vorher male:", accuracy_male)
print(f'Recall: {recall_score(y_test_s_male, y_pred_s_male)}')
confusionbefore_s_male = (confusion_matrix(y_test_s_male, y_pred_s_male))
print(confusionbefore_s_male)
# Ergebnis als Balkendiagramm ausgeben male
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [confusionbefore_s_male[0, 0], confusionbefore_s_male[0, 1], confusionbefore_s_male[1, 0], confusionbefore_s_male[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix swapped male')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.savefig('swapping_methodik_male.png', dpi=my_dpi * 2)
plt.show()

# evaluation female
df_test_s = pd.concat([X_test_s, y_test_s], axis = 1)
df_test_s_female = df_test_s[df_test_s['Gender']==1]
X_test_s_female = df_test_s_female.drop(['Hired'], axis=1)
y_test_s_female = df_test_s_female['Hired']
y_pred_s_female = xgb_model_s.predict(X_test_s_female)
# Genauigkeit berechnen
accuracy_s_female = accuracy_score(y_test_s_female, y_pred_s_female)
print("Genauigkeit vorher female:", accuracy_s_female)
print(f'Recall: {recall_score(y_test_s_female, y_pred_s_female)}')
confusionbefore_s_female = (confusion_matrix(y_test_s_female, y_pred_s_female))
print(confusionbefore_s_female)
# Ergebnis als Balkendiagramm ausgeben female
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [confusionbefore_s_female[0, 0], confusionbefore_s_female[0, 1], confusionbefore_s_female[1, 0], confusionbefore_s_female[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix swapped female')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.savefig('swapping_methodik_female.png', dpi=my_dpi * 2)
plt.show()




# SMOTE
df_smote = df.copy()
smote = SMOTE(random_state=42)
X = df.drop('Hired', axis=1)
y = df['Hired']
# Gesamten Datensatz in Trainings- und Testdaten aufteilen
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X, y, test_size=0.2, random_state=42)
# SMOTE Methode auf Trainingsdaten anwenden
X_train_smote, y_train_smote = smote.fit_resample(X_train_smote, y_train_smote)

# Mean Difference
df_diff_smote = pd.concat([X_train_smote, y_train_smote], axis = 1)
mean_male = df_diff_smote[df_diff_smote['Gender'] == 0]['Hired'].mean()
mean_female = df_diff_smote[df_diff_smote['Gender'] == 1]['Hired'].mean()
mean_diff = mean_female - mean_male
print('Mean difference:', mean_diff)

X_train_smote, X_val_smote, y_train_smote, y_val_smote = train_test_split(X_train_smote, y_train_smote, test_size=0.2, random_state=42)
# XGBoost-Modelle trainieren
xgb_model_smote = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric=['auc'])
xgb_model_smote.fit(X_train_smote, y_train_smote, eval_set=[(X_val_smote, y_val_smote)])
# Vorhersagen für Testdatensatz durchführen
y_pred_smote = xgb_model_smote.predict(X_test_smote)

# Genauigkeit, Recall und Confusion Matrix für das Modell berechnen
smote_accuracy = accuracy_score(y_test_smote, y_pred_smote)
smote_recall = recall_score(y_test_smote, y_pred_smote)
smote_confusion = confusion_matrix(y_test_smote, y_pred_smote)
# Ergebnisse ausgeben
print("Genauigkeit nach SMOTE:", smote_accuracy)
print("Recall nach SMOTE:", smote_recall)
print("Confusion Matrix nach SMOTE:", smote_confusion)
# Ergebnis als Balkendiagramm ausgeben
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [smote_confusion[0, 0], smote_confusion[0, 1], smote_confusion[1, 0], smote_confusion[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.title('Confusion Matrix after SMOTE')
plt.savefig('smote_methodik_all.png', dpi=my_dpi * 2)
plt.show()

# evaluation male
df_test_smote = pd.concat([X_test_smote, y_test_smote], axis = 1)
df_test_smote_male = df_test_smote[df_test_smote['Gender']==0]
X_test_smote_male = df_test_smote_male.drop(['Hired'], axis=1)
y_test_smote_male = df_test_smote_male['Hired']
y_pred_smote_male = xgb_model_smote.predict(X_test_smote_male)
# Genauigkeit berechnen
accuracy_smote_male = accuracy_score(y_test_smote_male, y_pred_smote_male)
print("Genauigkeit vorher male:", accuracy_smote_male)
print(f'Recall: {recall_score(y_test_smote_male, y_pred_smote_male)}')
confusionbefore_smote_male = (confusion_matrix(y_test_smote_male, y_pred_smote_male))
print(confusionbefore_smote_male)
# Ergebnis als Balkendiagramm ausgeben male
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [confusionbefore_smote_male[0, 0], confusionbefore_smote_male[0, 1], confusionbefore_smote_male[1, 0], confusionbefore_smote_male[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix swapped male')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.savefig('smote_methodik_male.png', dpi=my_dpi * 2)
plt.show()

# evaluation female
df_test_smote = pd.concat([X_test_smote, y_test_smote], axis = 1)
df_test_smote_female = df_test_smote[df_test_smote['Gender']==1]
X_test_smote_female = df_test_smote_female.drop(['Hired'], axis=1)
y_test_smote_female = df_test_smote_female['Hired']
y_pred_smote_female = xgb_model_smote.predict(X_test_smote_female)
# Genauigkeit berechnen
accuracy_smote_female = accuracy_score(y_test_smote_female, y_pred_smote_female)
print("Genauigkeit vorher female:", accuracy_smote_female)
print(f'Recall: {recall_score(y_test_smote_female, y_pred_smote_female)}')
confusionbefore_smote_female = (confusion_matrix(y_test_smote_female, y_pred_smote_female))
print(confusionbefore_smote_female)
# Ergebnis als Balkendiagramm ausgeben male
fig, ax = plt.subplots()
categories = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [confusionbefore_smote_female[0, 0], confusionbefore_smote_female[0, 1], confusionbefore_smote_female[1, 0], confusionbefore_smote_female[1, 1]]
percentages = [f'{c/sum(counts)*100:.1f}%' for c in counts]
ax.barh(categories, counts)
ax.set_xticks(range(0, max(counts)+1, 2))
ax.set_xlabel('Counts')
ax.set_ylabel('Categories')
ax.set_title('Confusion Matrix swapped female')
for i, v in enumerate(counts):
    ax.text(v + 0.1, i, percentages[i], va='center')
plt.savefig('smote_methodik_female.png', dpi=my_dpi * 2)
plt.show()
