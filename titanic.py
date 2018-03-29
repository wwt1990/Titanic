# Load packages
import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import seaborn as sns
sns.set_style('whitegrid')


from sklearn.ensemble import RandomForestClassifier


# Load and check data
train_df = pd.read_csv('/Users/tian/Documents/Kaggle/Titanic/train.csv')
test_df = pd.read_csv('/Users/tian/Documents/Kaggle/Titanic/test.csv')
print('----------------------------------------')
train_df.info()
print('----------------------------------------')
test_df.info()
full = pd.concat([train_df, test_df], ignore_index=True)
print('----------------------------------------')
full.info()
full.describe()

full.describe(include =['O'])
full.describe(include = 'all')


# 1. Feature Engineering
# 1.1 "Name" analysis
full['Title'] = full['Name'].str.split(', ').str[1].str.split('.').str[0]

full[full.Title == 'Dona'].values[0]
full[full.Title == 'Dona'].Title.item()

survived_table = pd.crosstab(index = full['Survived'], columns = 'count')
sex_title = pd.crosstab(index = full['Sex'], columns = full['Title'], margins=True)
survived_title = pd.crosstab(index = full['Survived'], columns = full['Title'], margins=True)
survived_title.index = ['died', 'survived', 'total']
survived_sex = pd.crosstab(index = full['Survived'], columns = full['Sex'], margins=True)
survived_sex.index = ['died', 'survived', 'total']
surv_sex_class = pd.crosstab(index=full["Survived"], columns=[full["Pclass"],full["Sex"]], margins=True)
surv_sex_class.index = ['died', 'survived', 'total']
surv_sex_class
surv_sex_class[2]
surv_sex_class[2]["female"]

rare_title = ['Capt','Col','Don','Dona','Dr','Jonkheer','Lady','Major','Rev','Sir','the Countess']
full.loc[full.Title == 'Mlle', 'Title'] = 'Miss'
full.loc[full.Title == 'Ms', 'Title'] = 'Miss'
full.loc[full.Title == 'Mme', 'Title'] = 'Mrs'
full.loc[full.Title.isin(rare_title), 'Title'] = 'Rare Title'
pd.crosstab(index = full['Survived'], columns = full['Title'], margins=True)
pd.crosstab(index = full['Sex'], columns = full['Title'])

full['Surname'] = full['Name'].str.split(',').str[0]
np.unique(full['Surname']).size
print("We have %d surnames." % (np.unique(full['Surname']).size))

full = full.drop('Name', axis = 1)

# 1.2 "Cabin" analysis
sum(full.Cabin.isnull()) # 1014 NaN
full['Deck'] = full.Cabin.str[0]

full = full.drop('Cabin', axis = 1)

# 2 Missing Values
# 2.1 "Embark" missing Values
index = full.index[full['Embarked'].isnull()]  # 61, 829
full.loc[full['Embarked'].isnull()]

embark_fare = full.loc[(full['PassengerId'] != 62) & (full['PassengerId'] != 830)]
fig3 = plt.figure()
fig3.add_subplot(111)
ax3 = sns.boxplot(x = 'Embarked', y = 'Fare', hue = 'Pclass', data = embark_fare)
plt.hlines(80, -100, 100, colors = 'red', linestyles = 'dashed')
#ax3.yaxis.get_majorticklocs()
plt.show()

full.loc[(full['PassengerId'] == 62) | (full['PassengerId'] == 830), 'Embarked'] = 'C'

# 2.2 "Fare" missing values
full.loc[full['Fare'].isnull()]
fare_c3_S = full.loc[(full['Pclass'] == 3) & (full['Embarked'] == 'S')]
fare_median = fare_c3_S['Fare'].median()

fig4 = plt.figure()
fig4.add_subplot(111)
sns.violinplot(x = 'Fare', data = fare_c3_S)
plt.vlines(fare_median, -100, 100, colors = 'red', linestyles = 'dashed')
plt.show()

full.loc[full['PassengerId'] == 1044, 'Fare'] = fare_median

# 2.3 "Age" missing values
sum(full.Age.isnull()) # 263 NaN

index = full.index[full['Age'].isnull()]



median0 = full.loc[full['Title'] == 'Master'].median()['Age']
full['Age'][(np.isnan(full['Age'])) & (full['Title'] == 'Master')] = median0


median1 = full.loc[(full['Pclass'] == 1) & (full['Sex'] == 'female')].median()['Age']
full['Age'][(np.isnan(full['Age'])) & (full['Pclass'] == 1) & (full['Sex'] == 'female')] = median1

median2 = full.loc[(full['Pclass'] == 1) & (full['Sex'] == 'male')].median()['Age']
full['Age'][(np.isnan(full['Age'])) & (full['Pclass'] == 1) & (full['Sex'] == 'male')] = median2

median3 = full.loc[(full['Pclass'] == 2) & (full['Sex'] == 'female')].median()['Age']
full['Age'][(np.isnan(full['Age'])) & (full['Pclass'] == 2) & (full['Sex'] == 'female')] = median3

median4 = full.loc[(full['Pclass'] == 2) & (full['Sex'] == 'male')].median()['Age']
full['Age'][(np.isnan(full['Age'])) & (full['Pclass'] == 2) & (full['Sex'] == 'male')] = median4

median5 = full.loc[(full['Pclass'] == 3) & (full['Sex'] == 'female')].median()['Age']
full['Age'][(np.isnan(full['Age'])) & (full['Pclass'] == 3) & (full['Sex'] == 'female')] = median5

median6 = full.loc[(full['Pclass'] == 3) & (full['Sex'] == 'male')].median()['Age']
full['Age'][(np.isnan(full['Age'])) & (full['Pclass'] == 3) & (full['Sex'] == 'male')] = median6

full.groupby('Survived').agg({'Fare':
                                 {'Median': 'median',
                                  'Mean': 'mean', },

                             'Sex': {'Male': lambda x: (x == 'male').sum(),
                                     'Female': lambda x: (x == 'female').sum(), },

                             'Pclass': {'1': lambda x: (x == 1).sum(),
                                        '2': lambda x: (x == 2).sum(),
                                        '3': lambda x: (x == 3).sum(), },

                             'SibSp': {'Mean': 'mean', },
                             'Parch': {'Mean': 'mean', },

                             'Embarked': {'S': lambda x: (x == 'S').sum(),
                                          'C': lambda x: (x == 'C').sum(),
                                          'Q': lambda x: (x == 'Q').sum()},

                             'Age': {'Median': 'median',
                                     'Mean': 'mean', },
                             }
                            )


# 3. Prediction
imp_vars = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Title', 'Surname', 'Deck', 'Ticket']
train = full[:891]
test = full[891:]
X_train = train[imp_vars]
y_train = train['Survived'].astype(int)
X_test = test[imp_vars].copy()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in X_test.columns.values:
       # Encoding only categorical variables
       if X_test[col].dtypes == 'object':
           # Using whole data to form an exhaustive list of levels
           data = X_train[col].append(X_test[col])
           le.fit(data.values)
           X_train[col] = le.transform(X_train[col])
           X_test[col] = le.transform(X_test[col])



X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)




# 3 Random Forests
rf = RandomForestClassifier(
    n_estimators=1000, min_samples_split=10, min_samples_leaf=1,
    max_features='auto', max_leaf_nodes=None, oob_score=True, n_jobs=1, random_state=1
    )
rf.fit(X_train, y_train)
rf.score(X_train, y_train)

plt.plot(rf.feature_importances_)
print("%.4f" % rf.oob_score_)


n_est = [100, 200, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1500, 2000, 2500, 3000]
n_est = [1100, 1200, 1300, 1400, 1500, 1600]
n_est = [1200, 1250, 1300, 1350, 1400]
for n_e in n_est:
    rf = RandomForestClassifier(
        n_estimators=n_e, min_samples_split=10, min_samples_leaf=1,
        max_features='auto', max_leaf_nodes=None, oob_score=True, n_jobs=1, random_state=42
    )
    rf.fit(X_train, y_train)
    print('%d: %f' % (n_e, rf.oob_score_))

# n_estimators = 1300!
for n_s in range(2, 20):
    rf = RandomForestClassifier(
        n_estimators=1300, min_samples_split=n_s, min_samples_leaf=1,
        max_features='auto', max_leaf_nodes=None, oob_score=True, n_jobs=1, random_state=42
    )
    rf.fit(X_train, y_train)
    print('%d: %f' % (n_s, rf.oob_score_))

# min_samples_split = 10!

for n_l in range(1,20):
    rf = RandomForestClassifier(
        n_estimators=1300, min_samples_split=10, min_samples_leaf=n_l,
        max_features='auto', max_leaf_nodes=None, oob_score=True, n_jobs=1, random_state=42
    )
    rf.fit(X_train, y_train)
    print('%d: %f' % (n_s, rf.oob_score_))

# min_samples_leaf = 1!

#  final model
rf = RandomForestClassifier(
    n_estimators=1300, min_samples_split=10, min_samples_leaf=1,
    max_features='auto', max_leaf_nodes=None, oob_score=True, n_jobs=1, random_state=42
)
rf.fit(X_train, y_train)


rf.score(X_train, y_train)
# 0.93265993265993263



feas = list(rf.feature_importances_)
dic = dict(zip(imp_vars, feas))


sorted(dic.items(), key=lambda x: x[1], reverse=True)


y_pred = rf.predict(X_test)
Survied_pred = pd.DataFrame({'Survived': y_pred.astype(int)})
PassengerId = test['PassengerId'].reset_index(drop=True)
surv = pd.concat([PassengerId, Survied_pred], axis = 1)
surv.to_csv('survivalPrediction2.csv', index = False)
