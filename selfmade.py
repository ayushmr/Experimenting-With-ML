import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import scipy.sparse

mndata= MNIST('datasets')
in_img,in_lbl=mndata.load_training()
out_img,out_label=mndata.load_testing()
image_i,label_in=np.array(in_img,float),np.array(in_lbl,float)
image_in=1.0/(1.0+np.exp(-image_i))
###
thita=np.random.rand(784,10)
##returns 60000*10 array of expected values
def y_maker(label_1):
    y=scipy.sparse.csr_matrix ((np.ones((label_1.shape[0])),(np.array(range(60000)),label_1)),shape=(60000,10))
    y=np.array(y.todense())   
    return y   

#a=y_maker(label_in)
#print(label_in[5])
#takes 1 row of x 1*784 and overall thita matrix of 784*10 and gives 1*10 vector of probablities
def softmax(x, var):
    soft=np.exp(np.dot(x,var)-np.median(np.dot(x,var)))
    
    #print(np.dot(x,var),"\n\n\n",np.dot(x,var)-np.median(np.dot(x,var)),"\n\n\n",soft,np.sum(soft),"\n\n\n")
    soft_max=(soft/np.sum(soft)).reshape(1,10)
    return soft_max
#def update(x,vars,y):

#a=softmax(image_in[9].reshape(1,784),thita)
#print(a,"\n\n\n",np.sum(a),thita,"\n\n\n",label_in[9])
#returns loss scalar, 784*10 grad matrix, input 1*784,1*10,784*10

def getloss(x,y,w):
    #a=np.dot(x,w)#1*10
    xt=x.reshape(784,1)
    loss=-1/784*np.sum((y.T*np.log(softmax(x,w))))   
    grad=-1/784*np.dot(x.T,(y-softmax(x,w)))
    return loss, grad



def get_accuracy(out_img,out_label):
    m=len(out_label)
    out_lab=np.array(out_label,float)
    out_image=np.array(out_img,float)
    sum=0
    for i in range (int(m)):
        print(get_pred(out_image[i]),out_lab[i])
        if(float(get_pred(out_image[i]))==(out_lab[i])):
            sum=sum+1
            #print(sum)
        else:
            continue
    print(sum/m*100)
    return sum

def get_pred(x):
    X=x.reshape(1,784)
    y=softmax(X,w)
    return (np.argmax(y))


w=thita
print(w,"\n\n\n")
n=0.6
y_mat= y_maker(label_in)
for i in range(60000):
    x=image_in[i].reshape(1,784)
    y=y_mat[i].reshape(1,10)
    loss,grad=getloss(x,y,w)
    w=w-grad*n
    #print(w[1])
print(w)    
c= get_accuracy(out_img,out_label)





    











  

