`localized_narratives.py` creates a .json object from the localized narratives captions (https://google.github.io/localized-narratives/)that is accessible by the name of an image from ADE20K.
Images that have no caption provided from localized narratives are moved into a `no_captions` folder.

The .json contains the name of an image and the corresponding caption.

`split_captions.py` creates a .json-file from localized narratives where the captions for each image are split up into individual sentences. 
The split up captions are generated with ... 
#TODO cite Raunak


