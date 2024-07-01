Hello everybody!

This is Alan, I am happy to present you this novel automated multi tissue segmentation fro fetal brain MRIs.        
It is very important to undersand that in order to use this model we will need to do the following ...

1. Use first the Prep_data_CSF_7.py this allow us covert the Fetal Brain MRIs inputs into NumPy arrays, otherwise our training will require decades.
2. Then you will have to train the model Training_CSF_7.py view by view (axi, sag, cor).
3. To finalize use model_prediction.py, use a normal MRI and the output will be the predicted segmentation.

This Network is a U-Net, with Multi-View-Aggregation and Test-Time Augmentation...
We used a dice loss function. We probably will have better results, with focalized loss functions.
A total of 100 SUBJECTS from 5 different institutes coonform our training data-set.

The input size image need to be 192x192.

