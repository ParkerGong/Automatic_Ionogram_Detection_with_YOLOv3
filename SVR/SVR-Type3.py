
#!/usr/bin/python
# -*- coding:utf-8 -*-
 
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from PIL import Image
 
if __name__ == "__main__":
    '''
    N = 50
    np.random.seed(0)
    x = np.sort(np.random.uniform(0, 6, N), axis=0)
    y = 2*np.sin(x)
    x = x.reshape(-1, 1)
    print ('x =\n', x)
    print ('y =\n', y)
    '''
        #############################################################################
    
    # 图片二值化
    
    img = Image.open('Z_SWGO_I_59140_20180923020000_O_INSD_DIS(cutF).jpg')
     
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
    
    #X=[x for y in matrix2 for x in y]
    #矩阵降维
    X=matrix2.reshape(-1)
    print(len(X))
                
    
    for j in range(height1):
        for i in range(width1):
            if X[i+width1*j]==0:
                Xa.append([i])
                m=height1 - j
                Ya.append([m])
                
    Xa1=np.array(Xa,dtype=float)
    Ya1=np.array(Ya,dtype=float)
    plt.plot(Xa1, Ya1, 'mo', markersize=2)
    #############################################################################
    # 数据
    x = np.array(Xa1)
    y = np.array(Ya1)
    print ('SVR - RBF')
    svr_rbf = svm.SVR(kernel='rbf', gamma=0.01, C=20)
    svr_rbf.fit(x, y)
    '''
    print ('SVR - Linear')
    svr_linear = svm.SVR(kernel='linear', C=100)
    svr_linear.fit(x, y)
    '''
    '''
    print ('SVR - Polynomial')
    svr_poly = svm.SVR(kernel='poly', degree=2, C=100)
    svr_poly.fit(x, y)
    '''
    print ('Fit OK.')
 
    # 思考：系数1.1改成1.5
    x_test = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_rbf = svr_rbf.predict(x_test)
    #y_linear = svr_linear.predict(x_test)
    #y_poly = svr_poly.predict(x_test)
    #显示RBF 拟合结果
    #plt.plot(x_test, y_rbf, 'r-', linewidth=2, label='RBF Kernel')
    
    plt.figure(figsize=(9, 8), facecolor='w')
    plt.plot(x_test, y_rbf, 'r-', linewidth=2, label='RBF Kernel')
    #plt.plot(x_test, y_linear, 'g-', linewidth=2, label='Linear Kernel')
    #plt.plot(x_test, y_poly, 'b-', linewidth=2, label='Polynomial Kernel')
    #plt.plot(x, y, 'mo', markersize=6)
    plt.scatter(x[svr_rbf.support_], y[svr_rbf.support_], s=1, c='r', marker='*', label='RBF Support Vectors')
    plt.legend(loc='lower left')
    plt.title('SVR', fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()