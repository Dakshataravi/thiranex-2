
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

df = pd.read_csv('data/sample_dataset.csv')

X = df[['Age','Income']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

acc = accuracy_score(y_test, pred)

with open('output/model_metrics.txt', 'w') as f:
    f.write(f'Accuracy: {acc:.2f}')

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.savefig('output/confusion_matrix.png')
plt.close()

probs = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title('ROC Curve')
plt.savefig('output/roc_curve.png')
plt.close()

imp = pd.Series(rf.feature_importances_, index=X.columns)

plt.figure(figsize=(5,4))
imp.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.savefig('output/feature_importance.png')
plt.close()

print('Project executed successfully')
