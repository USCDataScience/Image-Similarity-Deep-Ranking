## ImageSimilarity DeepRanking
Deep Ranking based ImageSimilarity will be developed as plugin on ImageSpace.
(https://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf)

### Installation 
### Pre-requisites and Install

Libraries -
```
pip3 install urllib3
pip3 install numpy
pip3 install https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl
pip3 install torchvision
pip3 install skimage
pip3 install pandas
```
#### Downloading Triplets
The training of DeepRanking requires triplets images (query, positive and negative image) which can be found at [here](https://sites.google.com/site/imagesimilaritydata/download). Download ```QUERY_AND_TRIPLETS.TXT``` and store it in a folder.

#### Preparing dataset
After downloading ```QUERY_AND_TRIPLETS.TXT``` run ```retrieve_folder.py``` in same folder. This process will take about 30 minutes.

#### triplets.txt
Run triplet_sampler.py and also specify arguments.

``` python triplet_sampler.py --input_directory ./ --output_directory ./ --num_pos_images 10 --num_neg_images 10 ```

### Credits
ImageSpace is developed by the JPL-Kitware team funded through the DARPA Memex program. 

### Authors
1. Dr. Chris Mattmann, JPL.
2. Purvak Lapsiya, USC.

### Current Progress:
- [x] retrieve_folder.py - Implemented python script to retrieve triplets from query_and_triplets.txt and store it in different folders.
- [x] folder.py - If you have downloaded entire dataset without using retrieve_folder.py and want to add images to folder use this simple script.
- [x] deep_ranking.py - Implemented model P from paper using PyTorch. (few changes to be made)
- [x] triplet_sampler.py - Implemented a simple randomized sampler to make different triplets out of existing 5033 triplets.

### Work remaining -
- [ ] Completing deep_ranking.py to implement entire architecture and hindge loss.
- [ ] DataLoader - Implement a dataloader to pass triplets to model.
- [ ] distance.py - To calculate distance between embeddings of images.


### License
This project is licensed under the [Apache License, version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

