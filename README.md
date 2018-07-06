# Shadow Detector

## Installing pymeanshift

```
cd pymeanshift
./setup.py install
```
## Generating dataset

```
./make_dataset.py
```

## Training the model

```
./train.py
```

This will save the model along with the weights at ```model.h5``` after the
final iteration of training and also saves the models with the best accuracy
under the ```checkpoints``` directory.

## Detecting shadows

```
./detect_shadows.py --image <path to image> --model <path to model>
```

```<path to image>``` is the path to any image.

```<path to model>``` is the to either ```model.h5``` or any of the models under
the ```checkpoints``` directory.
