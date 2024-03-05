# Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework

## Installing Dependencies

```bash
pip install torch torchvision
pip install tensorboard thop spikingjelly==0.0.0.0.12
```

## Usage

To reproduce the experiments on CIFAR10 in the paper, simply follow the default settings

```bash
python main.py
```

You can specify the output path and the weight of penalty term $\lambda$ by

```bash
python main.py --penalty-lmbda <lambda> --output-dir <path>
```

To reproduce the experiments on other datasets, follow the settings in the appendix.

## Citation

```bibtex
@inproceedings{shi2024towards,
  title={Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework},
  author={Shi, Xinyu and Ding, Jianhao and Hao, Zecheng and Yu, Zhaofei},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
