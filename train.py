from feature import NPDFeature
from ensemble import AdaBoostClassifier
import numpy as np 
from pylab import *
from PIL import Image
import math
# import math
from sklearn import metrics
from sklearn import tree
def  trainX():
		#构建人脸数据
	dir_name = 'datasets/original/face/face_'
	im_list = []
	for i in range(0,100):
	    i = str(i).zfill(3) 
	    im = Image.open(dir_name + i + '.jpg').convert('L')
	    im = im.resize((24,24))
	    im = array(im)
	    im_list.append(im)
	# im_list
		#提取特征
	NF = NPDFeature(im_list[0])
	feature = NF.extract()
	f_list = feature
	for i in range(1,100):
	    NF = NPDFeature(im_list[i])
	    feature = NF.extract()
	    f_list = append(f_list,feature)
	f_list = f_list.reshape(100,165600)
	#构建非人脸数据   
	dir_name = 'datasets/original/nonface/nonface_'
	imn_list = []
	for i in range(0,50):
	    i = str(i).zfill(3) 
	    im = Image.open(dir_name + i + '.jpg').convert('L')
	    im = im.resize((24,24))
	    im = array(im)
	    imn_list.append(im)
	# imn_list
		#调用NPDFeatureclass提取特征
	NF = NPDFeature(imn_list[0])
	feature = NF.extract()
	fn_list = feature
	print(feature.shape)
	for i in range(1,50):
	    NF = NPDFeature(imn_list[i])
	    feature = NF.extract()
	    fn_list = append(fn_list,feature)
	fn_list = fn_list.reshape(50,165600)
		#将数据合并成训练集
	X = concatenate((f_list,fn_list),axis=0)
	return X

def testX():
	dir_name = 'datasets/original/face/face_'
	im_list_test = []
	for i in range(100,150):
	    i = str(i).zfill(3) 
	    im = Image.open(dir_name + i + '.jpg').convert('L')
	    im = im.resize((24,24))
	    im = array(im)
	    im_list_test.append(im)
	# im_list_test
	#提取特征
	NF = NPDFeature(im_list_test[0])
	feature = NF.extract()
	f_list_test = feature
	for i in range(1,50):
	    NF = NPDFeature(im_list_test[i])
	    feature = NF.extract()
	    f_list_test = append(f_list_test,feature)
	f_list_test = f_list_test.reshape(50,165600)
	#构建测试集的非人脸特征
	dir_name = 'datasets/original/nonface/nonface_'
	imn_list_test = []
	for i in range(50,75):
	    i = str(i).zfill(3) 
	    im = Image.open(dir_name + i + '.jpg').convert('L')
	    im = im.resize((24,24))
	    im = array(im)
	    imn_list_test.append(im)
	imn_list_test
	#提取特征
	NF = NPDFeature(imn_list_test[0])
	feature = NF.extract()
	fn_list_test = feature
	for i in range(1,25):
	    NF = NPDFeature(imn_list_test[i])
	    feature = NF.extract()
	    fn_list_test = append(fn_list_test,feature)
	fn_list_test = fn_list_test.reshape(25,165600)
	X_test = concatenate((f_list_test,fn_list_test),axis=0)
	return X_test
def dataY():
	y1 = np.ones(100)
	y2 = -np.ones(50)
	y_train = append(y1,y2)
	y_train = y_train.reshape(150,1)
	y1 = np.ones(50)
	y2 = -np.ones(25)
	y_test = append(y1,y2)
	y_test = y_test.reshape(75,1)
	return y_train,y_test
def acc(y_test,y_preds):
	for n,y in enumerate(y_preds):
	    if y > 0 :
	        y_preds[n] = 1
	    if y <= 0:
	        y_preds[n] = -1
	num = 0
	for z in zip(y_preds,y_test):
	    if int(z[0]) == int(z[1][0]):
	        num = num + 1
	print('arr：',num/len(y_test))

if __name__ == "__main__":
	X = trainX()
	X_test = testX()
	y_train,y_test = dataY()
	clf = tree.DecisionTreeClassifier(max_depth=50,min_samples_leaf=50,random_state=30,criterion='gini')
	gbdt = AdaBoostClassifier(clf,10)
	gbdt.fit(X,y_train)
	y_preds = gbdt.predict(X_test)
	# y_preds
	acc(y_test,y_preds)

