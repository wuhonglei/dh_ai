from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils.common import exists_cache, load_cache, save_cache
from sklearn.model_selection import train_test_split

data = load_cache('cache/sg_embeddings.pkl')
X = data['embeddings']
y = data['Category']
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(classification_report(y_test, preds))
