# Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework

## Installing Dependencies

```bash
pip install torch torchvision
pip install tensorboard thop spikingjelly==0.0.0.0.12
```

## Usage

To reproduce the experiments on CIFAR10 in the paper, use the default settings

```bash
python main.py
```

You may specify the data path, the output path, and the weight of penalty term $\lambda$ by

```bash
python main.py --penalty-lmbda <lambda> --data-path <path-to-your-data> --output-dir <path>
```

To reproduce the experiments on CIFAR10-DVS

```bash
python main.py -b 64 --T 10 --epoch-search 240 --epoch-finetune 80 --model VGGSNN --dataset CIFAR10DVS --augment --search-lr 0.025 --prune-lr 0.001 --finetune-lr 0.0025 --optimizer SGD --prune-optimizer Adam --criterion CE --search-lr-scheduler Cosine --finetune-lr-scheduler Cosine --TET --penalty-lmbda <lambda> --data-path <path-to-your-data> --output-dir <path>
```

To reproduce the experiments on ImageNet

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py -b 64 --T 4 --epoch-search 280 --epoch-finetune 40 --model sew_resnet18 --dataset ImageNet --search-lr 0.001 --finetune-lr 0.0001 --criterion CE --search-lr-scheduler Cosine --finetune-lr-scheduler Cosine --penalty-lmbda <lambda> --data-path <path-to-your-data> --output-dir <path>
```

## Citation

```bibtex
@inproceedings{shi2024towards,
  title={Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework},
  author={Shi, Xinyu and Ding, Jianhao and Hao, Zecheng and Yu, Zhaofei},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
