# deepfake-detection

## Solution description 
In general solution is based on frame-by-frame classification approach. 


#### Face-Detector
MTCNN detector is chosen due to kernel time limits.

### Input size
I have gone through a lot of reseach papers and  I discovered that EfficientNets significantly outperform other encoders I used only them in my solution.
As I started with B4 I decided to use "native" size for that network (380x380).
Due to memory costraints I did not increase input size even for B7 encoder.

### Margin
When I generated crops for training I added 30% of face crop size from each side and used only this setting during the competition. 


### Encoders
The winning encoder is current state-of-the-art model (EfficientNet B7) pretrained with ImageNet and noisy student [Self-training with Noisy Student improves ImageNet classification
](https://arxiv.org/abs/1911.04252)

### Averaging predictions
I used 32 frames for each video.
For each model output instead of simple averaging I used the following heuristic which worked quite well on public leaderbord (0.25 -> 0.22 solo B5).
```python
import numpy as np

def confident_strategy(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    # 11 frames are detected as fakes with high probability
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)
```

### Augmentations

I used heavy augmentations by default. 
[Albumentations](https://github.com/albumentations-team/albumentations) library supports most of the augmentations out of the box.

def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )
``` 


## Data preparation

##### 1. Find face bboxes
To extract face bboxes I used facenet library, basically only MTCNN. 


##### 2. Extract crops from videos
To extract image crops I used bboxes saved before. It will use bounding boxes from original videos for face videos as well.

 
##### 3. Generate landmarks
From the saved crops it is quite fast to process crops with MTCNN and extract landmarks  

 
##### 4. Generate diff SSIM masks


##### 5. Generate folds



## Training

Training 5 B7 models with different seeds.

During training checkpoints are saved for every epoch.


### Fake detection articles  
- [The Deepfake Detection Challenge (DFDC) Preview Dataset](https://arxiv.org/abs/1910.08854)
- [Deep Fake Image Detection Based on Pairwise Learning](https://www.mdpi.com/2076-3417/10/1/370)
- [DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection](https://arxiv.org/abs/2001.03024)
- [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](https://arxiv.org/abs/2001.00179)
- [Real or Fake? Spoofing State-Of-The-Art Face Synthesis Detection Systems](https://arxiv.org/abs/1911.05351)
- [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/abs/1912.11035)
- [FakeSpotter: A Simple yet Robust Baseline for Spotting AI-Synthesized Fake Faces](https://arxiv.org/abs/1909.06122)
- [FakeLocator: Robust Localization of GAN-Based Face Manipulations via Semantic Segmentation Networks with Bells and Whistles](https://arxiv.org/abs/2001.09598)
- [Media Forensics and DeepFakes: an overview](https://arxiv.org/abs/2001.06564)
- [Face X-ray for More General Face Forgery Detection](https://arxiv.org/abs/1912.13458)




