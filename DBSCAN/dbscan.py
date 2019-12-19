# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:28:30 2019

@author: 公园
"""

import numpy as np  # 数据结构
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics   # 评估模型
from sklearn import svm
import matplotlib.pyplot as plt  # 可视化绘图
from PIL import Image
from scipy.interpolate import spline


# 图片二值化
    
img = Image.open('test1_without_denoise(cut).jpg')
 
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
#print(len(X))
            

for j in range(height1):
    for i in range(width1):
        if X[i+width1*j]==0:
            Xa.append([i])
            m=height1 - j
            Ya.append([m])
            
Xa1=np.array(Xa,dtype=float)
Ya1=np.array(Ya,dtype=float)
#plt.plot(Xa1, Ya1, 'mo', markersize=2)######绘制原始图片

#data1=list(zip(Xa,Ya))
data2=np.hstack((Xa,Ya))





'''
data=[
    [-2.68420713,1.469732895],[-2.71539062,-0.763005825],[-2.88981954,-0.618055245],[-2.7464372,-1.40005944],[-2.72859298,1.50266052],
    [-2.27989736,3.365022195],[-2.82089068,-0.369470295],[-2.62648199,0.766824075],[-2.88795857,-2.568591135],[-2.67384469,-0.48011265],
    [-2.50652679,2.933707545],[-2.61314272,0.096842835],[-2.78743398,-1.024830855],[-3.22520045,-2.264759595],[-2.64354322,5.33787705],
    [-2.38386932,6.05139453],[-2.6225262,3.681403515],[-2.64832273,1.436115015],[-2.19907796,3.956598405],[-2.58734619,2.34213138],
    [1.28479459,3.084476355],[0.93241075,1.436391405],[1.46406132,2.268854235],[0.18096721,-3.71521773],[1.08713449,0.339256755],
    [0.64043675,-1.87795566],[1.09522371,1.277510445],[-0.75146714,-4.504983795],[1.04329778,1.030306095],[-0.01019007,-3.242586915],
    [-0.5110862,-5.681213775],[0.51109806,-0.460278495],[0.26233576,-2.46551985],[0.98404455,-0.55962189],[-0.174864,-1.133170065],
    [0.92757294,2.107062945],[0.65959279,-1.583893305],[0.23454059,-1.493648235],[0.94236171,-2.43820017],[0.0432464,-2.616702525],
    [4.53172698,-0.05329008],[3.41407223,-2.58716277],[4.61648461,1.538708805],[3.97081495,-0.815065605],[4.34975798,-0.188471475],
    [5.39687992,2.462256225],[2.51938325,-5.361082605],[4.9320051,1.585696545],[4.31967279,-1.104966765],[4.91813423,3.511712835],
    [3.66193495,1.0891728],[3.80234045,-0.972695745],[4.16537886,0.96876126],[3.34459422,-3.493869435],[3.5852673,-2.426881725],
    [3.90474358,0.534685455],[3.94924878,0.18328617],[5.48876538,5.27195043],[5.79468686,1.139695065],[3.29832982,-3.42456273]
]
'''
X = data2

db = skc.DBSCAN(eps=6, min_samples=10).fit(X) #DBSCAN聚类方法 还有参数，matric = ""距离计算方法
labels = db.labels_  #和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声

print('每个样本的簇标号:')
print(labels)

raito = len(labels[labels[:] == -1]) / len(labels)  #计算噪声点个数占总数的比例
print('噪声比:', format(raito, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

print('分簇的数目: %d' % n_clusters_)
print("轮廓系数: %0.3f" % metrics.silhouette_score(X, labels)) #轮廓系数评价聚类的好坏

'''
#################散点图平滑处理#################
cluster_db = X[labels == 1]
cluster_x = [i[0] for i in cluster_db]# 从a中的每一行取第一个元素。
cluster_y = [i[1] for i in cluster_db]

cx=np.array(cluster_x)
cy=np.array(cluster_y)

x_new = np.linspace(cx.min(),cx.max(),5) #300 represents number of points to make between T.min and T.max
y_smooth = spline(cx,cy,x_new)
plt.plot(x_new,y_smooth,c='red')
'''
#################均值滤波（此处为取聚类结果为'1'的进行试验）###############
cluster_db = X[labels == 1]
cluster_x = [i[0] for i in cluster_db]# 从a中的每一行取第一个元素。
cluster_y = [i[1] for i in cluster_db]

counter1=0
j=1
yrange = 0
clen=len(cluster_x)
xm=[]
xm1=[]
ym=[]
ym1=[]
'''
for i in range(clen):
    xrange=cluster_x[i]
    for j in range(len(cluster_x)):
        if cluster_x[j] == xrange:
            yrange += cluster_y[j]
            counter1 = counter1 + 1
    yrange_mean=yrange/counter1
    xm=xm+[xrange]
    ym=ym+[yrange_mean]
plt.plot(xm,ym)
'''    

cx=np.array(cluster_x)
cy=np.array(cluster_y)
j=cx.min()
for j in range(cx.min(),(cx.max()+1)):
    yrange = 0
    counter1 = 0
    for i in range(len(cluster_x)):
        if cluster_x[i] == j:
            yrange = yrange + cluster_y[i]
            counter1= counter1 + 1
    if counter1:
        yrange_mean=yrange / counter1
        xm=xm+[j]
        xm1.append([j])
        ym=ym+[yrange_mean]
        ym1.append([yrange_mean])
plt.plot(xm,ym)
####调用smooth平滑滤波####        
x_new = np.linspace(cx.min(),cx.max(),40) #300 represents number of points to make between T.min and T.max
y_smooth = spline(xm,ym,x_new)
plt.plot(x_new,y_smooth,c='red')    

    
####################svr拟合均值滤波后曲线###############
xsvr = np.array(xm1,dtype=float)
ysvr = np.array(ym1,dtype=float)
print ('SVR - RBF')
svr_rbf = svm.SVR(kernel='rbf', gamma=0.001, C=100)
svr_rbf.fit(xsvr, ysvr)
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
x_test = np.linspace(xsvr.min(), xsvr.max(), 100).reshape(-1, 1)
y_rbf = svr_rbf.predict(x_test)
#y_linear = svr_linear.predict(x_test)
#y_poly = svr_poly.predict(x_test)
#显示RBF 拟合结果
#plt.plot(x_test, y_rbf, 'r-', linewidth=2, label='RBF Kernel')

plt.figure(figsize=(9, 8), facecolor='w')
plt.plot(x_test, y_rbf, 'r-', linewidth=2)
#plt.plot(x_test, y_linear, 'g-', linewidth=2, label='Linear Kernel')
#plt.plot(x_test, y_poly, 'b-', linewidth=2, label='Polynomial Kernel')
#plt.plot(x, y, 'mo', markersize=6)
#plt.scatter(xsvr[svr_rbf.support_], ysvr[svr_rbf.support_], s=1, c='r', marker='*', label='RBF Support Vectors')
plt.legend(loc='lower left')
plt.title('SVR', fontsize=16)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()


#############绘制聚类结果#############
for i in range(n_clusters_):
    print('簇 ', i, '的所有样本:')
    one_cluster = X[labels == i]
    print(one_cluster)
    plt.plot(one_cluster[:,0],one_cluster[:,1],'o')##########绘制聚类后结果



plt.show()
