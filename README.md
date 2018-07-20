# Shadow Detector

## Installing Dependencies 

```
pip install -r requirements.txt
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

```
./detect_shadows.py --image <path to image> --model <path to model>
```

```<path to image>``` is the path to any image.

```<path to model>``` is the to either ```model.h5``` or any of the models under
the ```checkpoints``` directory.
