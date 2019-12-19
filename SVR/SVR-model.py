# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:53:46 2019

@author: 公园
"""

from __future__ import division
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from PIL import Image
 
rng = np.random.RandomState(0)

#############################################################################

# 图片二值化

img = Image.open('20130408140700(cut).jpg')
 
# 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
Img = img.convert('L')
Img.save("test1.jpg")
 
# 自定义灰度界限，大于这个值为黑色，小于这个值为白色
threshold = 200
 
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)
 
# 图片二值化
photo = Img.point(table, '1')
#photo.save("test2.jpg")
#photo就是二值化后的图片
matrix1 = np.array(photo)
matrix2=matrix1.astype(int)
width1,height1 = img.size
i=j=0
Xa=[]
Ya=[]

X=[x for y in matrix2 for x in y]
print(len(X))
            

for j in range(height1):
    for i in range(width1):
        if X[i+width1*j]==0:
            Xa.append([i])
            Ya.append([j])
Xa1=np.array(Xa,dtype=float)
Ya1=np.array(Ya,dtype=float)
#############################################################################
# 生成随机数据
X = np.array(Xa1).reshape(1, -1)
y = np.array(Ya1).reshape(1, -1)
#X = 5 * rng.rand(10000, 1)
#y = np.sin(X).ravel()

# 在标签中对每50个结果标签添加噪声
 
#y[::50] += 2 * (0.5 - rng.rand(int(X.shape[0]/50)))
 
X_plot = width1

#############################################################################
#训练规模
#train_size = 100
#初始化SVR
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=2,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
#记录训练时间
t0 = time.time()
#训练
#svr.fit(X[:train_size], y[:train_size])
svr.fit(X, y)
svr_fit = time.time() - t0
 
t0 = time.time()
#测试
y_svr = svr.predict(Xa1)
svr_predict = time.time() - t0
#############################################################################
# 对结果进行显示
plt.scatter(X, y ,c='k', label='data', zorder=1)
#plt.hold('on')

plt.plot(X_plot, y_svr, c='r',
         label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
 
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()
 
plt.figure()
'''
##############################################################################
# 对训练和测试的过程耗时进行可视化
X = 5 * rng.rand(1000000, 1)
y = np.sin(X).ravel()
y[::50] += 2 * (0.5 - rng.rand(int(X.shape[0]/50)))
sizes = np.logspace(1, 4, 7)
for name, estimator in {
                        "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
    train_time = []
    test_time = []
    for train_test_size in sizes:
        t0 = time.time()
        estimator.fit(X[:int(train_test_size)], y[:int(train_test_size)])
        train_time.append(time.time() - t0)
 
        t0 = time.time()
        estimator.predict(X_plot[:1000])
        test_time.append(time.time() - t0)
 
    plt.plot(sizes, train_time, 'o-', color="b" if name == "SVR" else "g",
             label="%s (train)" % name)
    plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
             label="%s (test)" % name)
 
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Train size")
plt.ylabel("Time (seconds)")
plt.title('Execution Time')
plt.legend(loc="best")
################################################################################
# 对学习过程进行可视化
plt.figure()
 
svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
train_sizes, train_scores_svr, test_scores_svr = \
    learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                   scoring="neg_mean_squared_error", cv=10)
 
plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
         label="SVR")
 
plt.xlabel("Train size")
plt.ylabel("Mean Squared Error")
plt.title('Learning curves')
plt.legend(loc="best")
 
plt.show()
'''