from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
X, Y, = make_multilabel_classification(n_samples=500, n_features=300, n_classes=5, n_labels=5, allow_unlabeled=False,
                                       random_state=1)

# divide 80% 20% for training and test set
X_test, Y_test = X[400:], Y[400:]
X, Y = X[:400], Y[:400]


print len(X[0])
print Y

# Visualize plot
_, (ax1, ax2) = plt.subplots(1, 2, sharex='row', sharey='row')
_, (ax3, ax4) = plt.subplots(1, 2, sharex='row', sharey='row')
plt.subplots_adjust(bottom=.15)

X = PCA(n_components=2).fit_transform(X)

ax1.scatter(X[:, 0], X[:, 1], c=np.sum(Y * np.array([1, 2, 3, 4, 5]), axis=1))
ax1.set_title('Train set 5 labels 5 classes')
ax1.set_ylabel('Feature 1 count')
ax1.set_xlabel('Feature 0 count')

X_test = PCA(n_components=2).fit_transform(X_test)

ax2.set_title('Test labels')
ax2.scatter(X_test[:, 0], X_test[:, 1], c=np.sum(Y_test * np.array([1, 2, 3, 4, 5]), axis=1))
ax2.set_xlabel('Feature 0 count')

forest = RandomForestClassifier(n_estimators=100, random_state=1)
decision = DecisionTreeClassifier()

# training step
multi_target_R = MultiOutputClassifier(forest, n_jobs=-1)
result_R = multi_target_R.fit(X, Y)
result_R = multi_target_R.predict(X_test)
score_R = multi_target_R.score(X_test, Y_test)

multi_target_D = MultiOutputClassifier(decision, n_jobs=-1)
multi_target_D = multi_target_D.fit(X, Y)
result_D = multi_target_D.predict(X_test)
score_D = multi_target_D.score(X_test, Y_test)



# Plot classification result
ax3.scatter(X_test[:, 0], X_test[:, 1], c=np.sum(result_D * np.array([1, 2, 3, 4, 5]), axis=1))
ax3.set_title('Decision Tree labels %0.2f' % score_D)
ax3.set_ylabel('Feature 1 count')
ax3.set_xlabel('Feature 0 count')
X_w_D = []
for i in range(len(result_D)):
    if not np.array_equal(result_D[i], Y_test[i]):
        X_w_D.append(X_test[i])

if X_w_D != []:
    X_w_D = np.array(X_w_D, ndmin=2)
    ax3.scatter(X_w_D[:, 0], X_w_D[:, 1], c=(1, 0, 0,1), marker='x')


#Plot True - Wrong RandomForest Classification Result
ax4.set_title('Random forest %0.2f' % score_R)
ax4.scatter(X_test[:, 0], X_test[:, 1], c=np.sum(result_R * np.array([1, 2, 3, 4, 5]), axis=1))
ax4.set_xlabel('Feature 0 count')

X_w_R = []

for i in range(len(result_R)):
    if not np.array_equal(result_R[i], Y_test[i]):
        X_w_R.append(X_test[i])
if X_w_R != []:
    X_w_R = np.array(X_w_R, ndmin=2)
    ax4.scatter(X_w_R[:, 0], X_w_R[:, 1], c=(1, 0, 0,1), marker='x')

plt.show()
