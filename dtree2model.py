# coding: utf-8
from sklearn.ensemble import RandomForestClassifier

X1 = [[0, 0], [1, 1],[1, 0], [1, 0], [1, 0], [0, 0], [1, 1],[1, 0], [1, 0], [1, 0], [0, 0], [1, 1],[1, 0], [1, 0], [1, 0], [0, 0], [1, 1],[1, 0], [1, 0], [1, 0]]
y1 = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]

rfc = RandomForestClassifier()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.33, random_state=0)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# export CoreML Model

import coremltools
cmmodel = coremltools.converters.sklearn.convert(rfc)
cmmodel.save("DTs.mlmodel")
