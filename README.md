
# Unified Online Knowledge Distillation: Cold Start, Progressive Supervision and Mode-Oriented Temperature Scaling


## Requirements
* python3.6
* pytorch1.3.1
* cuda10.0
* lmdb

## Usages

**For CNN-liked Models**

To train the student and teacher model, please run the following command:
```
python ./UniOKD_code/Train/main.py
```

To evaluate the trained student or teacher model, please run the following command:
```
python ./UniOKD_code/Test/main.py
```

**For ViT-liked Models**

To train the student and teacher model, please run the following command:
```
python ./UniOKD_code_vit/ViT_res18-Swint/train.py
```






