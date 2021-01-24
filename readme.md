## Download the cropped dataset images
* [images.zip](https://drive.google.com/file/d/1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj) (355 MB)
images must be downloaded into directory named "images"  at the root of the repository. If you execute the python script (download_images.py), this step will be performed for you automatically.

## FileList
There are csv files that specify the train, validation and test pictures used for each scenario and fold.

## OrginalAndCroppedFileNames.xlsx 
The original names of the files of the images and the class information they belong to are found. 
   
## test_model.py
It is the script to be run for the prediction process. Parameter List:
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         'VGG16', 'VGG19', or 'ConcatedModel'.
  --scenario SCENARIO   'S1', 'S2','S3' or 'S4'.
  --fold FOLD           '1', '2','3','4' or '5'
  --imageName IMAGENAME 'Image Name'