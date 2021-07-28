`localized_narratives.py` creates a .json object from the localized narratives captions (https://google.github.io/localized-narratives/)that is accessible by the name of an image from ADE20K.
Images that have no caption provided from localized narratives are move into a `no_captions` folder.

The .json contains the name of an image and the corresponding caption.