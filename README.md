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
pip3 install tqdm
```
#### Downloading Triplets
The training of DeepRanking requires triplets images (query, positive and negative image) which can be found at [here](https://sites.google.com/site/imagesimilaritydata/download). Download ```QUERY_AND_TRIPLETS.TXT``` and store it in a folder.

#### Preparing dataset
After downloading ```QUERY_AND_TRIPLETS.TXT``` run ```retrieve_folder.py``` in same folder. This process will take about 30 minutes.
This python code will parse through txt file -- retrieve triplets and store it in folder. It'll repeat this process for 5033 Triplets.

#### triplets.txt
Run triplet_sampler.py and also specify arguments.

``` python triplet_sampler.py --input_directory ./ --output_directory ./ --num_pos_images 10 --num_neg_images 10 ```

Triplet sampler is used here to augment dataset, since relevance score (mentioned in paper) is not publically available this technique is used. Sampler will pair two similar images (of 1 folder) with positive/negative images of another folder.

#### deep_ranking.py
deep_ranking.py models the architecture given in paper, current implementation just prints layers of P,Q,R network. Also, tripletLoss is preferred over hinge loss as it more suitable to our use case.

#### trained model 
You can find link to download trained model [here](https://drive.google.com/file/d/1TmUKqp_TnzSP0TeAHIyTv8jG4KZeNqQP/view?usp=sharing)

```python deep_ranking.py```

### Current Progress:
- [x] retrieve_folder.py - Implemented python script to retrieve triplets from query_and_triplets.txt and store it in different folders.
- [x] folder.py - If you have downloaded entire dataset without using retrieve_folder.py and want to add images to folder use this simple script.
- [x] deep_ranking.py - Implemented model P,Q,R from paper using PyTorch along with optimizer and loss function.
- [x] triplet_sampler.py - Implemented a simple randomized sampler to make different triplets out of existing 5033 triplets.

### Work remaining -
- [ ] DataLoader - Implement a dataloader to pass triplets to model.
- [ ] distance.py - To calculate distance between embeddings of images.

### Credits
ImageSpace is developed by the JPL-Kitware team funded through the DARPA Memex program. 

### Authors
1. Dr. Chris Mattmann, JPL.
2. Purvak Lapsiya, USC.

### License
This project is licensed under the [Apache License, version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

