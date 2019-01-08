import os
import io
import urllib
from urllib.request import urlretrieve

triplets=[]
def download_filess(triplets_url):
    names = ['query','positive','negative']
    index = 0
    flag=0
    file_dir = './'
    for triplet in triplets_url:
        file_path = file_dir+str(index)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        else:
            print('path exists for triplet: ',index)
        final_path = []
        for image,name in zip(triplet,names):
            try:
                urllib.request.urlretrieve(image, file_path+'/'+name+".jpg")
                final_path.append(file_path+'/'+name+".jpg")
            except:
                flag=1
        if flag == 0:
            triplets.append(final_path)
        flag = 0
        index+=1


def dataset_loader(query):
    f = open(query,'r+')
    length = len(f.readlines())

    with open(query,'r+') as fp:
        triplet_return = []
        index = 0
        for j in range(length//4):
            triplet_return.append([])
            for i in range(0,4):
                if i == 0:  
                    _ = fp.readline()
                else:
                    ans = fp.readline()
                    ans = ans[:-1]
                    triplet_return[index].append(ans)
                if i==3:
                    index+=1
        return triplet_return


if __name__ == "__main__":

    triplets_url = dataset_loader("query_and_triplets.txt")
    #print(triplets_url[:10])
    download_filess(triplets_url)
