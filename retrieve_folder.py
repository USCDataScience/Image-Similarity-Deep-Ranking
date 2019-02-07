import os
import io
import urllib
from urllib.request import urlretrieve
from tqdm import tqdm

#global list to track triplets 
triplets=[]

def download_files(triplets_url):
    
    '''
    Retrieves triplets given in triplets_url using urlib/urlib3 and store it in "file_dir".
    It stores by creating separate folder for each triplet pair in file_dir.
    '''
    
    names = ['query','positive','negative']
    index = 0
    flag=0
    file_dir = './'
    for triplet in tqdm(triplets_url):
        file_path = file_dir+str(index)
        
        #To check if path exists with given "file_path" name.
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        else:
            print('path exists for triplet: ',index)
        
        final_path = []
        for image,name in zip(triplet,names):
            
            #Some triplets are not publically available thereby try, catch.
            try:
                urllib.request.urlretrieve(image, file_path+'/'+name+".jpg")
                final_path.append(file_path+'/'+name+".jpg")
            except:
                flag=1
                
        if flag == 0:
            triplets.append(final_path)
        flag = 0
        index+=1
        if(index%100==0 and index>=100):
            print(index,"triplets added..")


def dataset_loader(query):
    
    '''
    Prepares data for download_files by parsing through 'query_and_triplets.txt' and storing triplet pairs as list.
    '''
    
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
    download_files(triplets_url)
    print("Triplets downloaded!")
