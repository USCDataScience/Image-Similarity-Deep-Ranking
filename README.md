## Image-Similarity-Deep-Ranking
Implementing Deep ranking based image similarity (https://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf)
Advisor - Dr. Chris Mattmann.

Current Progress:
1. retrieve_folder.py - Implemented python script to retrieve triplets from query_and_triplets.txt and store it in different folders.
2. folder.py - If you have downloaded entire dataset without using retrieve_folder.py and want to add images to folder use this simple script.
3. deep_ranking.py - Implemented model P from paper using PyTorch. (few changes to be made)
4. triplet_sampler.py - Implemented a simple randomized sampler to make different triplets out of existing 5033 triplets[yet to be added].

Work remaining -
1. Completing deep_ranking.py to implement entire architecture and hindge loss.
2. DataLoader - Implement a dataloader to pass triplets to model.
3. distance.py - To calculate distance between embeddings of images.
