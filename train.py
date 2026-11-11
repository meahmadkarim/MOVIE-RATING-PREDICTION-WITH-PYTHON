import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
p=argparse.ArgumentParser(); p.add_argument('--data',default='IMDb Movies India.csv'); p.add_argument('--trees',type=int,default=100); args=p.parse_args()
df=pd.read_csv(args.data,encoding='latin1').dropna(subset=['Rating']).copy()
df['Year']=pd.to_numeric(df.Year.astype(str).str.extract(r'(\d{4})',expand=False),errors='coerce')
numeric=['Year']; categorical=['Genre','Director','Actor 1','Actor 2','Actor 3']
X_train,X_test,y_train,y_test=train_test_split(df[numeric+categorical],df.Rating,test_size=.2,random_state=42)
pre=ColumnTransformer([('numeric',SimpleImputer(strategy='median'),numeric),('category',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Unknown')),('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
model=Pipeline([('preprocess',pre),('model',RandomForestRegressor(n_estimators=args.trees,random_state=42,n_jobs=1))]).fit(X_train,y_train)
pred=model.predict(X_test); baseline=DummyRegressor(strategy='mean').fit(X_train,y_train)
print(json.dumps({'test_rows':len(y_test),'mae':mean_absolute_error(y_test,pred),'rmse':mean_squared_error(y_test,pred)**.5,'baseline_mae':mean_absolute_error(y_test,baseline.predict(X_test))},indent=2))
