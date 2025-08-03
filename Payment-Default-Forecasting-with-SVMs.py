import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder, scale, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')

df = pd.read_excel('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/eWtLeiKCjP9dCyP9AecgPA/default%20of%20credit%20card%20clients.xls')
print(df.head())

new_header = df.iloc[0]
df = df[1:] 
df.columns = new_header
df.rename({'default payment next month': 'DEFAULT'}, axis='columns', inplace=True)
df.drop('ID', axis=1, inplace=True)
print(df.head())

# check dimensions for invalid values
print(df['GENDER'].unique())
print(df['MARRIAGE'].unique())
print(df['EDUCATION'].unique())
print(df['AGE'].unique())

# count missing or null values
print(len(df[pd.isnull(df.GENDER)]))
print(len(df[pd.isnull(df.MARRIAGE)]))
print(len(df[pd.isnull(df.EDUCATION)]))
print(len(df[pd.isnull(df.AGE)]))

#count of missing data
print(len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)])) 
#Filter the DataFrame
df_no_missing_data = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]

# Explore distribution of data set
# count plot on ouput variable
ax = sns.countplot(x = df_no_missing_data['DEFAULT'], palette = 'rocket')

#add data labels
ax.bar_label(ax.containers[0])

# add plot title
plt.title("Observations by Classification Type")

# show plot
plt.show()
# split data
df_no_default = df_no_missing_data.loc[(df_no_missing_data['DEFAULT']==0)]
df_default = df_no_missing_data.loc[(df_no_missing_data['DEFAULT']==1)]

# downsample the data set
df_no_default_downsampled = resample(df_no_default, replace=False, n_samples=1000, random_state=42 )
df_default_downsampled = resample(df_default, replace=False, n_samples=1000, random_state=42 )

#check ouput
print(len(df_no_default_downsampled))
print(len(df_default_downsampled))

# merge the data sets
df_downsample = pd.concat([df_no_default_downsampled, df_default_downsampled ])
print(len(df_downsample))

# isolate independent variables
X = df_downsample.drop('DEFAULT', axis=1).copy()

ohe = OneHotEncoder(sparse_output=False, dtype="int")
ohe.fit(X[['GENDER', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']])
X_ohe_train = ohe.transform(X[['GENDER', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']])

X_ohe_train

transformed_ohe = pd.DataFrame(
    data=X_ohe_train,
    columns=ohe.get_feature_names_out(['GENDER', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']),
    index=X.index,
)
transformed_ohe.head()

# merge dataframes
X_encoded = pd.concat([X, transformed_ohe], axis=1)
X_encoded.head()

y = df_downsample['DEFAULT'].copy()
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

#scale the data
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit LabelEncoder to y_train and transform y_train
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

clf_svm = SVC(random_state = 42) # you will get the same result everytime you run
clf_svm.fit(X_train_scaled, y_train_encoded)

#calculate overall accuracy
y_pred = clf_svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2%}')


class_names = ['Did Not Default', 'Defaulted']
disp = ConfusionMatrixDisplay.from_estimator(
        clf_svm,
        X_test_scaled,
        y_test_encoded,
        display_labels=class_names,
        cmap=plt.cm.Blues)


param_grid = {'C':[0.5,0.1,1,10,100,1000],
              'gamma':['scale', 1,0.1, 0.01,0.001,0.0001],
              'kernel':['rbf']}

optimal_params = GridSearchCV(SVC(), param_grid, cv = 5, scoring='accuracy', verbose=3)
optimal_params.fit(X_train_scaled, y_train_encoded)

# see "best" parameters
optimal_params.best_params_

# refit model with optimal hyperparameters
grid_predictions = optimal_params.predict(X_test.values)
clf_svm = SVC(random_state = 42, C=.5, gamma=0.01)
clf_svm.fit(X_train_scaled, y_train_encoded)

#calculate overall accuracy
y_pred = clf_svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2%}')

# plot confusion matrix
class_names = ['Did Not Default', 'Defaulted']
disp = ConfusionMatrixDisplay.from_estimator(
        clf_svm,
        X_test_scaled,
        y_test_encoded,
        display_labels=class_names,
        cmap=plt.cm.Blues)
        
param_grid = {'C':[0.5,0.1,1,10,100,1000],
              'gamma':['scale', 1,0.1, 0.01,0.001,0.0001],
              'kernel':['rbf']}

optimal_params = GridSearchCV(SVC(), param_grid, cv = 5, scoring='accuracy', verbose=3)
optimal_params.fit(X_train_scaled, y_train_encoded)

# see "best" parameters
optimal_params.best_params_

# refit model with optimal hyperparameters
grid_predictions = optimal_params.predict(X_test.values)
clf_svm = SVC(random_state = 42, C=.5, gamma=0.01)
clf_svm.fit(X_train_scaled, y_train_encoded)

#calculate overall accuracy
y_pred = clf_svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Accuracy: {accuracy:.2%}')

# plot confusion matrix
class_names = ['Did Not Default', 'Defaulted']
disp = ConfusionMatrixDisplay.from_estimator(
        clf_svm,
        X_test_scaled,
        y_test_encoded,
        display_labels=class_names,
        cmap=plt.cm.Blues)

pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]

#plot scree plot
plt.bar(x=range(1, len(per_var)+1), height=per_var)
plt.tick_params(axis='x', which = 'both', bottom=False, top=False, labelbottom=False)
plt.ylabel("Explained variance (%)")
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.show()

train_pc1_coords = X_train_pca[:, 0]
train_pc2_coords = X_train_pca[:, 1]

pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

param_grid = {'C':[0.01, 0.1, 0.5, 1, 10, 100],
              'gamma':[1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
              'kernel':['rbf']}
optimal_params = GridSearchCV(SVC(), param_grid, cv = 5, scoring='accuracy', verbose=3)

optimal_params.fit(pca_train_scaled, y_train_encoded)

clf_svm = SVC(random_state=42, C=1000, gamma=0.001)
clf_svm.fit(pca_train_scaled, y_train_encoded)

X_test_pca = pca.transform(X_train_scaled)
test_pc1_coords = X_test_pca[:, 0]
test_pc2_coords = X_test_pca[:, 1]

x_min = test_pc1_coords.min()-1
x_max = test_pc1_coords.max()+1
y_min = test_pc2_coords.min()-1
y_max = test_pc2_coords.max()+1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),np.arange(start=y_min, stop=y_max, step=0.1) )

Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = Z.reshape(xx.shape)

# visualizing the data
fig, ax = plt.subplots(figsize=(10,10))
ax.contourf(xx,yy, Z, alpha=0.1)
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])
scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train_encoded, cmap=cmap, s=100, edgecolors='k', alpha=0.7)
legend = ax.legend(scatter.legend_elements()[0], scatter.legend_elements()[1], loc='upper right')
legend.get_texts()[0].set_text('Did Not Default')
legend.get_texts()[1].set_text('Defaulted')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Visualizing the Decision Boundary Using Principal Components')
plt.show()  