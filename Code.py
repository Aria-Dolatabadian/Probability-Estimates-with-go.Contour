import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
mesh_size = .02
margin = 0.25
# Load and split data
a = pd.read_csv("df1.csv")
b = pd.read_csv("df2.csv")

# Convert pandas.core.frame.DataFrame to numpy.ndarray
X = np.loadtxt('df1.csv', delimiter=',')
y = np.loadtxt('df2.csv', delimiter=',')

X, y = make_moons(noise=0.3, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y.astype(str), test_size=0.25, random_state=0)

# Create a mesh grid on which we will run our model
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Create classifier, run predictions on grid
clf = KNeighborsClassifier(15, weights='uniform')
clf.fit(X, y)
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)


# Plot the figure
fig = go.Figure(data=[
    go.Contour(
        x=xrange,
        y=yrange,
        z=Z,
        colorscale='RdBu'
    )
])
fig.show()
