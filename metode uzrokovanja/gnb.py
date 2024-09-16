import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import os
import numpy as np

def load_all_arff_from_directory(directory):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith(".arff"):
            data, meta = arff.loadarff(os.path.join(directory, filename))
            df = pd.DataFrame(data)
            dataframes.append((filename, df))
    return dataframes

datasets = load_all_arff_from_directory('/content')

model = GaussianNB()

def evaluate_model(X_train_resampled, y_train_resampled, X_test, y_test):
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = extract_f1_macro_avg(report)
    return accuracy, f1_macro

def extract_f1_macro_avg(report):
    return report['macro avg']['f1-score']

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

results = []

sampling_strategies_over_under = [0.2, 0.5, 0.75, 1.0]  
sampling_strategies_smote = [0.2, 0.5, 0.75, 1.0]  

for filename, df in datasets:
  
    df['Defective'] = df['Defective'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    df = df.dropna()

    df['Defective'] = df['Defective'].map({'Y': 1, 'N': 0})

    X = df.iloc[:, :-1]
    y = df['Defective']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for strategy in sampling_strategies_over_under:
        ros_scores = []
        rus_scores = []

        for _ in range(10): 
            try:
              
                ros = RandomOverSampler(sampling_strategy=strategy)
                X_ros, y_ros = ros.fit_resample(X_train, y_train)
                scores_ros = cross_val_score(model, X_ros, y_ros, cv=cv, scoring='f1_macro')
                ros_scores.append(scores_ros.mean())

                rus = RandomUnderSampler(sampling_strategy=strategy)
                X_rus, y_rus = rus.fit_resample(X_train, y_train)
                scores_rus = cross_val_score(model, X_rus, y_rus, cv=cv, scoring='f1_macro')
                rus_scores.append(scores_rus.mean())

                print(f"Filename: {filename}, Strategy: {strategy}")
                print(f"OverSampling Scores: {scores_ros}")
                print(f"UnderSampling Scores: {scores_rus}")

            except ValueError as e:
                print(f"Skipping {filename} with strategy {strategy} due to error: {e}")

        mean_ros = np.mean(ros_scores)
        std_ros = np.std(ros_scores)
        mean_rus = np.mean(rus_scores)
        std_rus = np.std(rus_scores)

        results.append({
            'Dataset': filename,
            'Sampling Strategy': strategy,
            'Original Class Distribution': y_train.value_counts().to_dict(),
            'Random OverSampling F1 Macro Avg': f"{mean_ros:.2f}±{std_ros:.2f}",
            'Random UnderSampling F1 Macro Avg': f"{mean_rus:.2f}±{std_rus:.2f}"
        })

    for strategy in sampling_strategies_smote:
        smote_scores = []

        for _ in range(10):  
            try:
             
                smote = SMOTE(sampling_strategy=strategy)
                X_smote, y_smote = smote.fit_resample(X_train, y_train)
                scores_smote = cross_val_score(model, X_smote, y_smote, cv=cv, scoring='f1_macro')
                smote_scores.append(scores_smote.mean())

                print(f"Filename: {filename}, SMOTE Strategy: {strategy}")
                print(f"SMOTE Scores: {scores_smote}")

            except ValueError as e:
                print(f"Skipping {filename} with SMOTE strategy {strategy} due to error: {e}")

        mean_smote = np.mean(smote_scores)
        std_smote = np.std(smote_scores)

        results.append({
            'Dataset': filename,
            'Sampling Strategy': strategy,
            'Original Class Distribution': y_train.value_counts().to_dict(),
            'SMOTE F1 Macro Avg': f"{mean_smote:.2f}±{std_smote:.2f}"
        })

results_df = pd.DataFrame(results)

results_df.to_csv('/content/f1_macro_avg_results.csv', index=False)

print(results_df)