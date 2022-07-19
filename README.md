# Neural Architecture Search for Spiking Neural Networks
Pytorch implementation code for [Neural Architecture Search for Spiking Neural Networks] (https://arxiv.org/abs/2201.10355)

For getting knowledge on NAS without training in ANN domain (refer: https://github.com/BayesWatch/nas-without-training)
## Prerequisites
* Python 3.9    
* PyTorch 1.10.0     
* NVIDIA GPU (>= 12GB)      
* CUDA 10.2 (optional)         

## Getting Started

### Conda Environment Setting
```
conda create -n SNASNet 
conda activate SNASNet
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install scipy
```
### Spikingjelly Installation (ref: https://github.com/fangwei123456/spikingjelly)
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install
```

## Training and testing

* Arguments required for training and testing are contained in ``config.py``` 
* Here is an example of running an experiment on CIFAR100
* (if a user want to skip search process and use predefined architecgtur) A architecture can be parsed by ``--cnt_mat 0302 0030 3003 0000`` format

Example) Architecture and the corresponding connection matrix

<img src="https://user-images.githubusercontent.com/41351363/142759748-50d0e9bf-4654-4831-97eb-5bfb4d30c21e.png"  width="630" height="400"/>


### Training

*  Run the following command

```
python search_snn.py  --exp_name 'cifar100_backward' --dataset 'cifar100'  --celltype 'backward' --batch_size 32 --num_search 5000 
```
simple argument instruction

--exp_name: savefile name

--dataset: dataset for experiment

--celltype: find backward connections or forward connections

--num_search: number of architecture candidates for searching

### Testing with pretrained models (CIFAR10 & CIFAR100)

(1) download pretrained parameters 

CIFAR10: ([link][e]) to ```./savemodel/save_cifar10_bw.pth.tar```   

[e]: https://drive.google.com/file/d/1pnS0nFMk2KlxTFeeVT5fYMdTPh_8qn84/view?usp=sharing

CIFAR100: ([link][e]) to ```./savemodel/save_cifar100_bw.pth.tar```   

[e]: https://drive.google.com/file/d/1pnS0nFMk2KlxTFeeVT5fYMdTPh_8qn84/view?usp=sharing

(2) The above pretrained model is for 

CIFAR10 architecture ``--cnt_mat 0303 0030 2002 0200``

CIFAR100 architecture ``--cnt_mat 0302 0030 3003 0000``

(3)  Run the following command
```
python search_snn.py  --dataset 'cifar10' --cnt_mat 0303 0030 2002 0200 --savemodel_pth './savemodel/save_cifar10_bw.pth.tar'  --celltype 'backward' --second_avgpooling 4
```
```
python search_snn.py  --dataset 'cifar100' --cnt_mat 0302 0030 3003 0000 --savemodel_pth './savemodel/save_cifar100_bw.pth.tar'  --celltype 'backward'
```

 

