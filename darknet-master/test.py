import os
def delet(path1,path2):
    filelist = os.listdir(path1)
    for item in filelist:
        item_path1=item
        if item.endswith('.jpg'): 
            item_path2=item[:-4]+'.xml'
            if not os.path.exists(item_path2):
                os.remove(item_path1)
                print(item_path1)
 
if __name__ == '__main__':
    inputimagePath = 'D:\Desktop\data' 
    inputlabel = 'D:\Desktop\data' 
   
    
    delet(inputimagePath,inputlabel)