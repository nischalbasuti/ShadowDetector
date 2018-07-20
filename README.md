# Shadow Detector

## Installing Dependencies 

```
pip install -r requirements.txt
git submodule update --init
cd pymeanshift
./setup.py install
```

## Downloading SBU-Dataset and setting up project structure

```
./init.sh
```

Running this will download the [SBU shadow dataset](http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip)
and extract it under the ```data``` directory, as well as create the
```checkpoints``` directory.

## Training the model

```
./train.py
```

This will save the models with the best accuracies under the ```checkpoints```
directory.

## Detecting shadows

To pass the whole image to the CNN and find the segments which are shadows:
```
./detect_shadows.py --image <path to image> --model <path to model>
```

To pass individual segments to the CNN and find the which are shadows:
```
./detect_shadows_by_segment.py --image <path to image> --model <path to model>
```

```<path to image>``` is the path to any image.

```<path to model>``` is the to either ```model.h5``` or any of the models under
the ```checkpoints``` directory.
