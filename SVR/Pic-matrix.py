from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
 
def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    width,height = im.size
    data = im.getdata()
    data = np.matrix(data,dtype='float')
    # 这里处理的是rgb三通道的图片
    #new_data = np.reshape(data.getA(),(height,width,3))
   # 单色图片：
    new_data = np.reshape(data,-1,height*width)
    return new_data
 
def MatrixToImage(data):
    # 显示图片
    #data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im
 
filename = './result.png'
data = ImageToMatrix(filename)
# print(data)
new_im = MatrixToImage(data)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()
