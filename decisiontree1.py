import pandas as pd
import numpy as np

dataset=pd.read_csv("data.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,5]

from sklearn.preprocessing import LabelEncoder #converts string into numbers
le=LabelEncoder()

x=x.apply(le.fit_transform)
print(x)

from sklearn.tree import DecisionTreeClassifier
regressor=DecisionTreeClassifier()
regressor.fit(x.iloc[:,1:5],y)
print(regressor)

x_in=np.array([1,1,0,0])
y_pred=regressor.predict([x_in])
print("Prediction:",y_pred)

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data=StringIO()

export_graphviz(regressor,out_file=dot_data,filled=True,rounded=True,special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("tree1.png")