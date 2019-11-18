import numpy as np
from train_n_test_nnet import train_nnet, test_nnet
from nnet_3layers import NNet

num_in, num_hide, num_out = 784, 200, 10
learning_rate = 0.2
num_epochs = 4
n = NNet(num_in, num_out, num_hide, learning_rate=learning_rate)
train_nnet(n, 'data\\mnist_train.csv', num_epochs)
print(test_nnet(n, 'data\\mnist_test.csv'))

# print(efficiency)
W_in_h, W_h_out = n.return_W()
np.save('W_in_h', W_in_h)
np.save('W_h_out', W_h_out)
n.load_W(np.load('W_in_h.npy'), np.load('W_h_out.npy'))
a, b = n.return_W()
