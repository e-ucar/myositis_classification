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
