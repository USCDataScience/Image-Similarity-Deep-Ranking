import io
import urllib.request 



triplets = []
path = './'

def download_dataset(triplets_url):
	names = ['query','positive','negative']
	for triplet in triplets_url:
		for image in triplet:
			urllib.request.urlretrieve(image, './test.jpg')
			break
		break




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



if __name__ == '__main__':
	triplets_url = dataset_loader("query_and_triplets.txt")
	print(len(triplets_url))
	download_dataset(triplets_url)
