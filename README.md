# Berkely_Capstone
## Pool Accuracy
With in this Jupyter Notebook, [main.ipynb](./main.ipynb), you will find steps that I have used to try to determine attributes that allow me to predict what pool a set of bins should be set to.

The types of pools that this classifier will try to predict is REG, EARLY, MID, and LATE. These pertain to the timeframe throughout the year that the Apples will be packed on a production line. An apple with good QC scores has the ability to withstand a controlled atmospheric room for up to 12 months to then be sent to a grocery store to be eaten by consumers.

## Analysis
The first step in the process is to make sure that we understand the data as much as possible and use various techniques to clean it and shape it into a format so that it is usable for predictions.

The first step for me was to clean up some of the null values that were within my dataset. Not all records within the set had QC records within them, however I knew which pool they got assigned to due to our packing records so I went ahead and took the averages of those pools and assigned that to those values.

```python
df['avg_size'] = df.groupby('pool')['avg_size'].transform(lambda x: x.fillna(x.mean()))
df['avg_temp'] = df.groupby('pool')['avg_temp'].transform(lambda x: x.fillna(x.mean()))
df['avg_weight'] = df.groupby('pool')['avg_weight'].transform(lambda x: x.fillna(x.mean()))
df['avg_starch'] = df.groupby('pool')['avg_starch'].transform(lambda x: x.fillna(x.mean()))
df['avg_pressure'] = df.groupby('pool')['avg_pressure'].transform(lambda x: x.fillna(x.mean()))
df['avg_split'] = df.groupby('pool')['avg_split'].transform(lambda x: x.fillna(x.mean()))
df['avg_watercore'] = df.groupby('pool')['avg_watercore'].transform(lambda x: x.fillna(x.mean()))
df['avg_frozen'] = df.groupby('pool')['avg_frozen'].transform(lambda x: x.fillna(x.mean()))
df['avg_tiacidity'] = df.groupby('pool')['avg_tiacidity'].transform(lambda x: x.fillna(x.mean()))
df['avg_lightexposure'] = df.groupby('pool')['avg_lightexposure'].transform(lambda x: x.fillna(x.mean()))
```

I also did similar steps with qccount, and also dropped storagetype.

## Preparation
To further prep the data for the models I compiled a definition of encoding and scaling the data so that it has the proper format to run through models. This definition will also go ahead and split away your X and y values that you will use for training and testing purposes.

```python
def encode(df):
    # Encode the target column 'y'
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['pool'])
    X = df.drop(columns=['pool'], axis=1)

    # Encode all categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['number']).columns

    # Encode categorical columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Encode numerical columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    return X, y
```

## Modeling
For the first phase of modeling I went through and created a DummyClassifier so that I could get a good baseline for what the data is saying as a good default threshold.

```python
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
baseline_acc = accuracy_score(y_test, dummy_pred)
```

This gave us a 31% accuracy score off the bat. This I believe is do to how the classifiers are split out within the dataset.
    ```
    pool
    REG      31.074897
    LATE     26.067719
    MID      22.511188
    EARLY    20.346196
    ```

The next stage was to run a LogisticRegression model. This though had consequences as the dataset that I have was too large to run efficiently so some adjustments had to be taken.

```python
X_trimmed = X[['cropyear', 'growerid', 'avg_size', 'avg_temp', 'avg_weight', 'avg_starch', 'avg_pressure', 'avg_split', 'avg_watercore', 'avg_frozen', 'avg_tiacidity', 'avg_lightexposure']]

# Add in all variety columns
variety_columns = [col for col in X.columns if 'variety_' in col]
X_trimmed = pd.concat([X_trimmed, X[variety_columns]], axis=1)
```

This forced me to only take into account all the QC data and then the variety columns.

You can then see the initial graph here from the run:
![Model Initial Performance](./images/initial_model_performance_comparison.png)

## Re-analysis
The next phase is to see how we can improve our accuracy against the initial testing through default models. The next approach is to try to utilize the functionality of GridSearch to loop through different parameters and determine which variation of a model will produce the greatest accuracy towards our desired result.

Through these adjustments I was able to produce roughly a 4% increase in accuracy but still well below my expectations of where it needs to be for a real world business application.

The best results were through a DT at 64% accuracy.

![Results](./images/results.png)

## Next steps
Clearly through this work the accuracies are still not there for me to be able to put this into a full application. The next phase it to pull in more of the production QC tests that happen after the apples are pulled out of the rooms to be able to get more QC data points that would give use a better determination what/if there are any coorelations.

After we get an acceptable success rate on pool classification, > 90%, then we want to start to layer in if we see any coorelation between the room itself and any degregation patterns in the QC scores of before and after.