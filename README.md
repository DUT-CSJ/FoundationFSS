# High-Performance Few-Shot Segmentation with Foundation Models: An Empirical Study

[arxiv link](https://arxiv.org/pdf/2409.06305)

### Datasets

Please follow [HSNet](https://github.com/juhongm999/hsnet?tab=readme-ov-file#preparing-few-shot-segmentation-datasets) to prepare few-shot segmentation datasets.

### Training and Testing

Training

```
python train.py --benchmark pascal/coco --logpath ./your_path --fold 0/1/2/3 --img_size 420
```

Testing

```
python test.py --benchmark pascal/coco --load ./your_path/model.pt --fold 0/1/2/3 --nshot 1/5 --img_size 420
```

### To Do

- [x] [Code](https://github.com/DUT-CSJ/FoundationFSS).
- [x] [Weights](https://github.com/DUT-CSJ/FoundationFSS/releases/download/weights/model_weights.zip).
- [x] [Training and testing commands](https://github.com/DUT-CSJ/FoundationFSS/blob/main/README.md#training-and-testing).
- [ ] Requirments.

### References

Thanks to [HSNet](https://github.com/juhongm999/hsnet), a great few-shot segmentation codebase!
