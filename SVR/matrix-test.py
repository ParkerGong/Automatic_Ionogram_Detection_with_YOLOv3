
# 图片二值化
from PIL import Image
import numpy
import matplotlib.pyplot as plt
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
photo.save("test2.jpg")
#photo就是二值化后的图片
matrix1 = numpy.array(photo)
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
            
Xa1=numpy.array(Xa,dtype=float)
Ya1=numpy.array(Ya,dtype=float)
            
plt.scatter(Xa, Ya ,c='k', label='data', zorder=1)  
