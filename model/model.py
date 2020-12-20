import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv("C:/titanic/train.csv")

# check dataset
print(df_train.head())
print(df_train.tail())
print(df_train.info())

# check NaN-included column
print(df_train.Age.isna())
print(df_train.Cabin.isna())
print(df_train.Embarked.isna())

# drop obvious NaN
# inplace => return current dataframe after function-completed
df_train.dropna(subset=["Embarked"], inplace=True)
print(df_train["Embarked"].value_counts())
print(df_train.info())

# fill NaN by mean value for numerical feature
df_train.fillna(df_train.mean(), inplace=True)
print(df_train.info())

# remove unusable column
df_train.drop(["PassengerId", "Cabin", "Name", "Ticket"], axis=1, inplace=True)
print(df_train.info())
print(df_train.head())

# convert from string to numeric
sex_mapping = {
    "female": 0,
    "male": 1
}

df_train["Sex"] = df_train["Sex"].map(sex_mapping)
print(df_train.head())
print(df_train["Survived"].value_counts())

# remove outlier value of numerical features by standardization z-value
df_train_without_outlier = df_train[np.abs(df_train["Age"] - df_train["Age"].mean()) <= 3 * np.std(df_train["Age"])]
print(df_train_without_outlier.info())
df_train_without_outlier = df_train[np.abs(df_train["Fare"] - df_train["Fare"].mean()) <= 3 * np.std(df_train["Fare"])]
print(df_train_without_outlier.info())
print(df_train_without_outlier.head())

# check correlation among all features => Fare is the most relative feature to Survived
corrMatt = df_train_without_outlier[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
corrMatt = corrMatt.corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False

fig, axes = plt.subplots()
fig.set_size_inches(20, 10)
sns.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)

# learning
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="lbfgs")
X_train = df_train_without_outlier[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
Y_train = df_train_without_outlier["Survived"]
model.fit(X_train, Y_train)

# evaluation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
scoring = "accuracy"
score = cross_val_score(model, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
mean_score = round(np.mean(score) * 100, 2)
print(mean_score)

# prediction
df_test = pd.read_csv("C:/titanic/test.csv")
print(df_test.head())
print(df_test.tail())

df_test["Sex"] = df_test["Sex"].map(sex_mapping)
df_test.fillna(df_test.mean(), inplace=True)
df_test.dropna(subset=["Fare"], inplace=True)
print(df_test.head())
print(df_test.info())

predict = model.predict(df_test.drop(["PassengerId", "Cabin", "Name", "Ticket", "Embarked"], axis=1))
result = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': predict
})
result.to_csv('C:/titanic/result.csv', index=False)

