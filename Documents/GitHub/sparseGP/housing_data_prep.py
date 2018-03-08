import numpy as np
from sklearn.preprocessing import scale

dat = np.loadtxt('./data/housing.data.txt')
train_id = np.random.choice(range(506), 455, replace=False)
test_id = np.setdiff1d(range(506), train_id)

x_mat = dat[:,0:13]
y_mat = dat[:,13]

x_train = x_mat[train_id, :]
x_test = x_mat[test_id, :]
y_train = y_mat[train_id]
y_test = y_mat[test_id]
mean_y = y_train.mean()
y_train = y_train - mean_y
y_test = y_test - mean_y

np.savetxt('./data/housing_train_x_nsc.txt', x_train, delimiter='\t')
np.savetxt('./data/housing_train_y_nsc.txt', y_train, delimiter='\t')
np.savetxt('./data/housing_test_x_nsc.txt', x_test, delimiter='\t')
np.savetxt('./data/housing_test_y_nsc.txt', y_test, delimiter='\t')