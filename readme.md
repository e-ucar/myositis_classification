## Download the cropped images of the dataset
* [images.zip](https://drive.google.com/file/d/1z5XCOOi0-Yz5UTbmKm6egRgW5ukVrUvf) (347 MB)
images must be downloaded into directory named "images"  at the root of the repository. If you execute the python script (download_images.py), this step will be performed for you automatically.

## FileList
There are csv files that specify the train, validation and test images used for each scenario and fold.

## OrginalAndCroppedFileNames.xlsx 
It is the file containing the original file names of the images in the dataset and the classes they belong to. 
   
## test_model.py
It is the script to be run for the prediction process. Parameter List:
optional arguments:<br />
  -h, --help            show this help message and exit. <br />
  --model MODEL         'VGG16', 'VGG19', or 'ProposedModel'. <br />
  --scenario SCENARIO   'S1', 'S2','S3' or 'S4'. <br />
  --fold FOLD           '1', '2','3','4' or '5'. <br />
  --imageName IMAGENAME 'Image Name'

## citation
@article{UCAR2022103277,
title = {Classification of myositis from muscle ultrasound images using deep learning},
journal = {Biomedical Signal Processing and Control},
volume = {71},
pages = {103277},
year = {2022},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2021.103277},
url = {https://www.sciencedirect.com/science/article/pii/S1746809421008740},
author = {Emine Uçar},
keywords = {Inflammatory myopathies, Classification, Deep learning},
abstract = {Inflammatory myopathies, are rare muscle diseases. As a result of the body's own immune system attacking by targeting the muscle cells, muscle weakness develops due to inflammation in the muscles. Early and definitive diagnosis of the disease is very important for treatment. In this study, it is aimed to develop a computer-aided diagnosis system that diagnoses diseases from muscle ultrasound images using deep learning methods. 3214 muscle ultrasound images of 19 inclusion body myositis, 14 polymyositis, 14 dermatomyositis and 33 normal patients were used as dataset. In the study, a new deep learning model was proposed by combining the VGG16 and VGG19 architectures which are well known in terms of classification performance. The proposed model has been tested on binary and multiple classification problems. For binary classification problems, in the first scenario (S1), normal images were examined with all disease images. In the second scenario (S2), normal images were examined with only inclusion body myositis images. In the third scenario (S3) inclusion body myositis images were examined with dermatomyositis and polymyositis images. The proposed model reached an average accuracy of 93.00% in S1, 96.01% in S2, 91.74% in S3 and 95.12% in multi-class classification (S4). The results obtained from the test dataset indicated that the proposed deep learning approach is effective in automatic classification of inflammatory myopathies.}
}
