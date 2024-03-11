# Road Damage Detection
-----
This repository contain codes for damage detection on the road.
In the city, it's important problem.
This project introduce in Oslo traffic center.

The solution is based on Deep CNN.

![](doc/bigdata21.png)


<img src="doc/road.JPG" width="350">





### Usage

Consider unisng a workspace for cleaner 

1. install required Libraries : 
1.1.
- python>= 3.6
- pytorch 1.4 or 1.6
- torchvision >= 0.5
- apex is also needed
- timm >= 1.28


`pip install -r requirements.txt` 

1.2.

install apex (2020-10-26): 

```
git clone https://github.com/NVIDIA/apex
pip install -r apex/requirements.txt
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```


2. download Dataset :
for ease of use we have provided annotations and ... in coco format downloadable: 
- [train and validation sets](https://drive.google.com/file/d/1IHaqAxpMtFwPHia7msB_1QPAywPgg7fW/view?usp=sharing)
- [test1 and test2 data](https://drive.google.com/uc?id=1apjJfNHUlKQS64IaHRg3qRp_T0NnZnnQ&export=download)(no annotation). 


otherwise one can download original data from sekilab github repo and convert using tools provided in utils folder.
 

3. for training : 

- Train on single GPU : 
```
python train.py ../data --model tf_efficientdet_d0 -b 40 --amp --lr .15 --sync-bn --opt fusedmomentum --warmup-epochs 3 --lr-noise 0.3 0.9 --model-ema --model-ema-decay 0.9998 -j 25 --epochs 300
```

- Distributed Training : (note you may need to make the file executable before training using `chmod +x distributed_train.sh`)

```
./distributed_train.sh 3 ../data --model tf_efficientdet_d0 -b 40 --amp --lr .15 --sync-bn --opt fusedmomentum --warmup-epochs 3 --lr-noise 0.3 0.9 --model-ema --model-ema-decay 0.9998 -j 25 --epochs 300 
```

4. for inference on testset and generating submission file :
```
python infer.py ./data --model tf_efficientdet_d0 --checkpoint ./path/to/model/checkpoint --use-ema --anno test1 -b 17 --threshold 0.300
```
5. Image Inference to generate detected images

- first create image_info_annotations(e.g. if image folder is in `../data` path. One should first create image info in json format using `python utils/createimageinfo.py` then folder structure should be like )

```
..
├── data
│   └── annotations
|       ├── image_info_test1.json
├── test1
│   ├── Japan_XXX.jpg
│   └── Czech_xxx.jpg
|   └── ....

```

following command will create generated file with bounding boxes in ./predictions
```
python detector.py ../data --model tf_efficientdet_d0 --checkpoint path.to/modelfile.pth.tar --anno test1  -b 20 --use-ema  --tosave ./predictions 
```

for validation (AP scores) and benchmarking with `cuda.Event()` use the following command : 
```
python validate.py ../data --model tf_efficientdet_d0 --checkpoint path/to/model/checkpoint.pth.tar --anno val  -b 20 --use-ema
```


