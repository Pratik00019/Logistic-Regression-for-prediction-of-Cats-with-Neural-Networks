import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
%matplotlib inline

def load_dataset():
    with h5py.File('/content/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('/content/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])


    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]
print ("Number of training examples: m_train = " ,str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x = train_set_x_flatten/255.
test_set_x=test_set_x_flatten/255

#Activation Function
def sigmoid(z):
  s=1/(1+np.exp(-z))
  return s

def initialize_with_zeros(dim):
  w=np.zeros((dim,1))
  b=0
  return w,b

def propagate(w,b,X,Y):
  m=X.shape[1]
  A=sigmoid(np.dot(w.T, X) + b)      

  cost = -1/m*(np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)))    
  # print(Y*np.log(A))
  # print((1-Y)*np.log(1-A))                               #used for analysis
  #print((Y*np.log(A) + (1-Y)*np.log(1-A)).shape)

  #####################Backward Propogate#######################

  dw = 1/m*(np.dot(X, ((A-Y).T)))
  db = 1/m*(np.sum(A-Y))
  grads = {"dw": dw,
             "db": db}   
  return grads,cost

def optimize(w, b, X, Y, num_iterations, learning_rate,print_cost = False):
  costs=[]
  for i in range(num_iterations):
    grads,cost=propagate(w,b,X,Y)
    dw=grads['dw']
    db=grads['db']

    w=w-learning_rate*dw
    b=b-learning_rate*db

    if(i%100==0):
      costs.append(cost)
    
    if(print_cost and i%100==0):
      print("Cost after iteration %i: %f" %(i,cost))

  params={"w":w,"b":b}
  grads = {"dw": dw,
             "db": db}

  return params,grads,costs

def predict(w,b,X):
  m = X.shape[1]
  Y_prediction = np.zeros((1,m))
  w = w.reshape(X.shape[0], 1)

  A=sigmoid(np.dot(w.T,X)+b)

  for i in range(A.shape[1]):
    
    if(A[0][i]>0.5):
      Y_prediction[0][i]=1
    else:
      Y_prediction[0][i]=0
    
  return Y_prediction

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

index = 2
plt.imshow(train_set_x_orig[index])
img_flatten=train_set_x_orig[index].reshape(1,-1).T
a=predict(d["w"],d["b"],img_flatten)
if(a[0][0]>0.6):
  print("It is a cat")
else:
  print("Not a cat")
