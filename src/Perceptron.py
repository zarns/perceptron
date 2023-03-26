import numpy as np

class Normalization:
    def __init__(self,):
        self.mean = np.zeros([1,64]) # means of training features
        self.std = np.zeros([1,64]) # standard deviation of training features

    def fit(self,x):
        self.mean = np.mean(x,axis=0)
        self.std = np.std(x,axis=0) 

    def normalize(self,x):
        # normalize the given samples to have zero mean and unit variance (add 1e-15 to std to avoid numeric issue)
        x = (x-self.mean)/(self.std+ 10**(-15)) #adding 1e-15 to prevent dividing by zero in fit()
        return x

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for i in range(len(label)):
        # print(label[i])
        one_hot[i][label[i]]=1
    return one_hot

def tanh(x):
    x = np.clip(x,a_min=-100,a_max=100) 
    f_x = np.tanh(x)
    # f_x = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return f_x

def softmax(x):
    # output should sum to one
    f_x = x
    f_x = np.exp(f_x)
    f_x = f_x / (10**(-15) + np.sum(f_x,axis=1))[:,np.newaxis]

    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0
        
        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters
            # update the parameters based on sum of gradients for all training samples

            z = tanh(self.bias_1 + np.matmul(train_x,self.weight_1))
            y = softmax(self.bias_2 + np.matmul(z,self.weight_2))

            part1 = np.matmul((train_y-y),self.weight_2.transpose())
            part2 = (1-z**2)
            part3 = np.multiply(part1,part2)
            part4 = np.matmul(train_x.transpose(),part3)

            v_grad = lr*np.matmul(z.transpose(),(train_y-y))
            w_grad = lr*part4
            b1_grad = lr*np.sum(part3,axis=0)
            b2_grad = lr*np.sum((y-train_y),axis=0)
            
            for i in range(len(b1_grad)):
                self.bias_1[0][i] = b1_grad[i]
            for i in range(len(b2_grad)):
                self.bias_2[0][i] = b2_grad[i]

            # print(b1_grad)
            # print("b1_grad" + str(b1_grad.shape))
            # print(self.bias_1)
            # print("bias1" + str(self.bias_1.shape))
            
            self.weight_1 += w_grad
            self.weight_2 += v_grad

            # part1 = np.matmul((train_y-y),self.weight_2.transpose())
            # # part2 = 1-np.matmul(z.transpose(),z)
            # part2 = np.matmul((1-z**2).transpose(),z)
            # part3 = np.matmul(part1,part2)
            # # print(part2.shape)
            # part4 = np.matmul(train_x.transpose(),part3)

            # w_grad = lr*part4
            # b1_grad = lr*np.sum(part3,axis=0)
            # b2_grad = lr*np.sum((y-train_y),axis=0)

            # dzdB = np.subtract(1,z**2)
            # part1 = np.matmul((train_y-y),self.weight_2.transpose())
            # part2 = np.matmul(z,dzdB.transpose())
            # part3 = np.matmul(part1.transpose(),part2)
            # part4 = np.matmul(part3,train_x)
            # w_grad = lr*part4.transpose()
            # b1_grad = lr*np.sum(part3.transpose(),axis=0)
            # b2_grad = lr*np.sum((y-train_y),axis=0)

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        z = tanh(self.bias_1 + np.matmul(x,self.weight_1))
        # convert class probability to predicted labels
        y = softmax(self.bias_2 + np.matmul(z,self.weight_2))
        ret = np.zeros([len(y),]).astype('int') # placeholder
        for i in range(len(y)):
            ret[i] = np.argmax(y[i])
        # print(ret[5])
        return ret

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        z = tanh(self.bias_1 + np.matmul(x,self.weight_1))
        # z = x # placeholder
        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
