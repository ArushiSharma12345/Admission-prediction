Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
data=pd.read_csv('C:\\Users\\acer\\Desktop\\vrtul\\Admission_Predict.csv')
data.head()
   Serial No.  GRE Score  TOEFL Score  ...  CGPA  Research  Chance of Admit 
0           1        337          118  ...  9.65         1              0.92
1           2        324          107  ...  8.87         1              0.76
2           3        316          104  ...  8.00         1              0.72
3           4        322          110  ...  8.67         1              0.80
4           5        314          103  ...  8.21         0              0.65

[5 rows x 9 columns]
data.tail()
     Serial No.  GRE Score  TOEFL Score  ...  CGPA  Research  Chance of Admit 
395         396        324          110  ...  9.04         1              0.82
396         397        325          107  ...  9.11         1              0.84
397         398        330          116  ...  9.45         1              0.91
398         399        312          103  ...  8.78         0              0.67
399         400        333          117  ...  9.66         1              0.95

[5 rows x 9 columns]
data.shape
(400, 9)
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 9 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Serial No.         400 non-null    int64  
 1   GRE Score          400 non-null    int64  
 2   TOEFL Score        400 non-null    int64  
 3   University Rating  400 non-null    int64  
 4   SOP                400 non-null    float64
 5   LOR                400 non-null    float64
 6   CGPA               400 non-null    float64
 7   Research           400 non-null    int64  
 8   Chance of Admit    400 non-null    float64
dtypes: float64(4), int64(5)
memory usage: 28.2 KB
data.isnull().sum()
Serial No.           0
GRE Score            0
TOEFL Score          0
University Rating    0
SOP                  0
LOR                  0
CGPA                 0
Research             0
Chance of Admit      0
dtype: int64
data.describe()
       Serial No.   GRE Score  ...    Research  Chance of Admit 
count  400.000000  400.000000  ...  400.000000        400.000000
mean   200.500000  316.807500  ...    0.547500          0.724350
std    115.614301   11.473646  ...    0.498362          0.142609
min      1.000000  290.000000  ...    0.000000          0.340000
25%    100.750000  308.000000  ...    0.000000          0.640000
50%    200.500000  317.000000  ...    1.000000          0.730000
75%    300.250000  325.000000  ...    1.000000          0.830000
max    400.000000  340.000000  ...    1.000000          0.970000

[8 rows x 9 columns]
data.columns
Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research', 'Chance of Admit '],
      dtype='object')
data=data.drop('Serial No.',axis=1)
data.columns
Index(['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '],
      dtype='object')
x=data.drop('Chance of Admit ',axis=1)
x
     GRE Score  TOEFL Score  University Rating  SOP  LOR   CGPA  Research
0          337          118                  4  4.5   4.5  9.65         1
1          324          107                  4  4.0   4.5  8.87         1
2          316          104                  3  3.0   3.5  8.00         1
3          322          110                  3  3.5   2.5  8.67         1
4          314          103                  2  2.0   3.0  8.21         0
..         ...          ...                ...  ...   ...   ...       ...
395        324          110                  3  3.5   3.5  9.04         1
396        325          107                  3  3.0   3.5  9.11         1
397        330          116                  4  5.0   4.5  9.45         1
398        312          103                  3  3.5   4.0  8.78         0
399        333          117                  4  5.0   4.0  9.66         1

[400 rows x 7 columns]
y=data['Chance of Admit ']
y
0      0.92
1      0.76
2      0.72
3      0.80
4      0.65
       ... 
395    0.82
396    0.84
397    0.91
398    0.67
399    0.95
Name: Chance of Admit , Length: 400, dtype: float64
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train
     GRE Score  TOEFL Score  University Rating  SOP  LOR   CGPA  Research
3          322          110                  3  3.5   2.5  8.67         1
18         318          110                  3  4.0   3.0  8.80         0
202        340          120                  5  4.5   4.5  9.91         1
250        320          104                  3  3.0   2.5  8.57         1
274        315          100                  1  2.0   2.5  7.95         0
..         ...          ...                ...  ...   ...   ...       ...
71         336          112                  5  5.0   5.0  9.76         1
106        329          111                  4  4.5   4.5  9.18         1
270        306          105                  2  2.5   3.0  8.22         1
348        302           99                  1  2.0   2.0  7.25         0
102        314          106                  2  4.0   3.5  8.25         0

[320 rows x 7 columns]
y_train
3      0.80
18     0.63
202    0.97
250    0.74
274    0.58
       ... 
71     0.96
106    0.87
270    0.72
348    0.57
102    0.62
Name: Chance of Admit , Length: 320, dtype: float64
data.head()
   GRE Score  TOEFL Score  University Rating  ...  CGPA  Research  Chance of Admit 
0        337          118                  4  ...  9.65         1              0.92
1        324          107                  4  ...  8.87         1              0.76
2        316          104                  3  ...  8.00         1              0.72
3        322          110                  3  ...  8.67         1              0.80
4        314          103                  2  ...  8.21         0              0.65

[5 rows x 8 columns]
from sklearn.preprocessing import StandardScaler
sc = 
KeyboardInterrupt
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=s
KeyboardInterrupt
x_test=sc.transform(x_test)
x_train
array([[ 0.45711129,  0.42466178, -0.057308  , ..., -1.05965163,
         0.13986648,  0.92761259],
       [ 0.1022887 ,  0.42466178, -0.057308  , ..., -0.50194025,
         0.36110014, -1.07803625],
       [ 2.05381293,  2.08593034,  1.6892215 , ...,  1.17119391,
         2.25009529,  0.92761259],
       ...,
       [-0.96217907, -0.40597251, -0.93057275, ..., -0.50194025,
        -0.62594237,  0.92761259],
       [-1.31700165, -1.40273364, -1.8038375 , ..., -1.61736302,
        -2.27668588, -1.07803625],
       [-0.25253389, -0.23984565, -0.93057275, ...,  0.05577114,
        -0.57488845, -1.07803625]])
data.head()
   GRE Score  TOEFL Score  University Rating  ...  CGPA  Research  Chance of Admit 
0        337          118                  4  ...  9.65         1              0.92
1        324          107                  4  ...  8.87         1              0.76
2        316          104                  3  ...  8.00         1              0.72
3        322          110                  3  ...  8.67         1              0.80
4        314          103                  2  ...  8.21         0              0.65

[5 rows x 8 columns]
from sklearn.linear_model import LinearRgression
Traceback (most recent call last):
  File "<pyshell#26>", line 1, in <module>
    from sklearn.linear_model import LinearRgression
ImportError: cannot import name 'LinearRgression' from 'sklearn.linear_model' (C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\linear_model\__init__.py)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# model training
lr=LinearRegression()
lr.fit(x_train,y_train)
LinearRegression()
svm=SVR()
svm.fit(x_train,y_train)
SVR()
rf=RandomForestRegressor()
ef.fit(x_train,y_train)
Traceback (most recent call last):
  File "<pyshell#37>", line 1, in <module>
    ef.fit(x_train,y_train)
NameError: name 'ef' is not defined. Did you mean: 'rf'?
rf..fit(x_train,y_train)
SyntaxError: invalid syntax
rf.fit(x_train,y_train)
RandomForestRegressor()
gr=GradientBoostingRegressor()
gr.fit(x_train,y_train)
GradientBoostingRegressor()
#prediction on test data
y_pred1=lr.predict(x_test)
y_pred2=svm.predict(x_test)
y_pred3=rf.predict(x_test)
y_pred4=gr.predict(x_test)
#Evaluate the algorithm
from sklearn import metrics
score1=metrics.r2_score(y_test,y_pred1)
score2=metrics.r2_score(y_test,y_pred2)
score3=metrics.r2_score(y_test,y_pred3)
score4=metrics.r2_score(y_test,y_pred4)
print(score1,score2,score3,score4)
0.8212082591486991 0.7597814848647668 0.8024103031714865 0.7969122571993726
final_data=pd.DataFrame({'Models':['LR','SVR','RF','GR'],R2_SCORE':
                         
SyntaxError: incomplete input
final_data=pd.DataFrame({'Models':['LR','SVR','RF','GR'],
                         'R2_SCORE':[score1,score2,score3,score4]})
                         
final_data
                         
  Models  R2_SCORE
0     LR  0.821208
1    SVR  0.759781
2     RF  0.802410
3     GR  0.796912
I#classification
                         
Traceback (most recent call last):
  File "<pyshell#58>", line 1, in <module>
    I#classification
NameError: name 'I' is not defined
#classification
                         
data.head()
                         
   GRE Score  TOEFL Score  University Rating  ...  CGPA  Research  Chance of Admit 
0        337          118                  4  ...  9.65         1              0.92
1        324          107                  4  ...  8.87         1              0.76
2        316          104                  3  ...  8.00         1              0.72
3        322          110                  3  ...  8.67         1              0.80
4        314          103                  2  ...  8.21         0              0.65

[5 rows x 8 columns]
import numpy as np
y_train=[1 if value>0.8 else 0 for value in y_train]
y_test=[1 if value>0.8 else 0 for value in y_test]
y_train=np.array(y_train)
y_test=np.array(y_test)
y_train
array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
       1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
       0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])
# import the model
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuract_score
Traceback (most recent call last):
  File "<pyshell#73>", line 1, in <module>
    from sklearn.metrics import accuract_score
ImportError: cannot import name 'accuract_score' from 'sklearn.metrics' (C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\metrics\__init__.py)
from sklearn.metrics import accuracy_score
# model training and evaluation
lr=LogisticRegression()
lr.fit(x_train,y_train)
LogisticRegression()
y_pred1=lr.predict(x_test)
print(accuracy_score(y_test,y_pred1))
0.925
svm=svm.SVC()
SVM.fit(x_train,y_train)
Traceback (most recent call last):
  File "<pyshell#81>", line 1, in <module>
    SVM.fit(x_train,y_train)
NameError: name 'SVM' is not defined. Did you mean: 'SVR'?
svm.fit(x_train,y_train)
SVC()
y_pred2=svm.predict(x_test)
print(accuracy_score(y_test,y_pred2))
0.925
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
KNeighborsClassifier()
y_pred3=knn.predict(x_test)
print(accuracy_score(y_test,y_pred3))
0.8875
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
RandomForestClassifier()
y_pred4=rf.predict(x_test)
print(accuracy_score(y_test,y_pred4))
0.9375
gr=GradientBoostingClassifier()
gr.fit(x_train,y_train)
GradientBoostingClassifier()
y_pred5=gr.predict(x_test)
print(accuracy_score(y_test,y_pred5))
0.975
final_data=pd.DataFrame({'Models':['LR','SVC','KNN','RF','GR'],
                         'ACC_SCORE':[accuracy_score(y_test,y_pred1),
                                      accuracy_score(y_test,y_pred2),
                                      accuracy_score(y_test,y_pred3),
                                      accuracy_score(y_test,y_pred4),
                                      accuracy_score(y_test,y_pred5)]})
final_data
  Models  ACC_SCORE
0     LR     0.9250
1    SVC     0.9250
2    KNN     0.8875
3     RF     0.9375
4     GR     0.9750
import seaborn as sns
sns.barplot(final_data['Models'],final_data['ACC_SCORE'])
Traceback (most recent call last):
  File "<pyshell#104>", line 1, in <module>
    sns.barplot(final_data['Models'],final_data['ACC_SCORE'])
TypeError: barplot() takes from 0 to 1 positional arguments but 2 were given
# save the model
data.columns
Index(['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '],
      dtype='object')
x = data.drop('Chance of Admit ',axis=1)
x
     GRE Score  TOEFL Score  University Rating  SOP  LOR   CGPA  Research
0          337          118                  4  4.5   4.5  9.65         1
1          324          107                  4  4.0   4.5  8.87         1
2          316          104                  3  3.0   3.5  8.00         1
3          322          110                  3  3.5   2.5  8.67         1
4          314          103                  2  2.0   3.0  8.21         0
..         ...          ...                ...  ...   ...   ...       ...
395        324          110                  3  3.5   3.5  9.04         1
396        325          107                  3  3.0   3.5  9.11         1
397        330          116                  4  5.0   4.5  9.45         1
398        312          103                  3  3.5   4.0  8.78         0
399        333          117                  4  5.0   4.0  9.66         1

[400 rows x 7 columns]
y=data['Chance of Admit ']
y
0      0.92
1      0.76
2      0.72
3      0.80
4      0.65
       ... 
395    0.82
396    0.84
397    0.91
398    0.67
399    0.95
Name: Chance of Admit , Length: 400, dtype: float64
y=[1 if value>0.8 else 0 for value in y]
y
[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]
y=np.array(y)
y
array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
       1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
       0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
       1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1,
       1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
       1, 1, 0, 1])
x= sc.fit_transform(x)
x
array([[ 1.76210664,  1.74697064,  0.79882862, ...,  1.16732114,
         1.76481828,  0.90911166],
       [ 0.62765641, -0.06763531,  0.79882862, ...,  1.16732114,
         0.45515126,  0.90911166],
       [-0.07046681, -0.56252785, -0.07660001, ...,  0.05293342,
        -1.00563118,  0.90911166],
       ...,
       [ 1.15124883,  1.41704229,  0.79882862, ...,  1.16732114,
         1.42900622,  0.90911166],
       [-0.41952842, -0.72749202, -0.07660001, ...,  0.61012728,
         0.30403584, -1.09997489],
       [ 1.41304503,  1.58200646,  0.79882862, ...,  0.61012728,
         1.78160888,  0.90911166]])
>>> gr=GradientBoostingClassifier()
>>> gr.fit(x,y)
GradientBoostingClassifier()
>>> import joblib
>>> joblib.dump(gr,""C:\\Users\\acer\\Desktop\\vrtul\\admission_model")
...             
SyntaxError: invalid syntax. Perhaps you forgot a comma?
>>> joblib.dump(gr,"C:\\Users\\acer\\Desktop\\vrtul\\admission_model")
...             
['C:\\Users\\acer\\Desktop\\vrtul\\admission_model']
>>> model=joblib.load('C:\\Users\\acer\\Desktop\\vrtul\\admission_model')
...             
>>> data.columns
...             
Index(['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research', 'Chance of Admit '],
      dtype='object')
>>> model.predict(sc.transform[[337,118,4,4.5,9.65,1]])
...             
Traceback (most recent call last):
  File "<pyshell#124>", line 1, in <module>
    model.predict(sc.transform[[337,118,4,4.5,9.65,1]])
TypeError: 'method' object is not subscriptable
>>> model.predict(sc.transform[[337,118,4,4.5,4.5,9.65,1]])
...             
Traceback (most recent call last):
  File "<pyshell#125>", line 1, in <module>
    model.predict(sc.transform[[337,118,4,4.5,4.5,9.65,1]])
TypeError: 'method' object is not subscriptable
>>> model.predict(sc.transform([[337,118,4,4.5,4.5,9.65,1]]))
...             

Warning (from warnings module):
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\base.py", line 409
    warnings.warn(
UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
array([1])
