import os
index = 3
file_num = 0
out=0
for filename in os.listdir('./'):
    file_path ='./'+str(file_num)
    if not os.path.exists(file_path):
        os.makedirs(str(file_num))    
    if filename.endswith(".jpg"):
        os.rename('./'+str(filename),'./'+str(file_num)+'/'+str(filename))
        index+=1
    if index ==3:
        file_num+=1
        index=0
