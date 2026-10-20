# ML-CNN-Imagenette-Classifier

A complete PyTorch implementation of the most basic CNN → MLP pipeline —
`Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → ReLU → Linear` —
**actually trained on real photographs until it can classify them**.

The dataset is **Imagenette** (10 classes pulled out of ImageNet, 9,469 training / 3,925 validation
real color photographs). On Apple Silicon (MPS), 12 epochs take about 12 minutes and reach
**70.14%** validation accuracy (random guessing = 10%).

> The point of this project is not to chase accuracy. It is to lay out, in numbers and pictures,
> the fact that with only two conv layers almost every parameter ends up in the first `Linear`
> after `Flatten` — and to record what you would change to push past that.

---

## Results at a glance

| Metric | Value |
|---|---|
| Validation accuracy (best = final, epoch 12) | **70.14%** |
| Training accuracy (epoch 12) | 73.20% |
| Total parameters (128×128 input) | 16,799,434 |
| ...of which `Linear(65536 → 256)` | 16,777,472 (**99.87%**) |
| Training time | 12 epochs / ~12 minutes (Apple Silicon MPS) |
| Environment | torch 2.10.0 + torchvision 0.25.0, device = `mps` |

---

## Architecture

```
input (N, 3, 128, 128)
  Conv2d(3 → 32, k=3, p=1)   → (N, 32, 128, 128)        896 params
  ReLU
  MaxPool2d(2)               → (N, 32,  64,  64)
  Conv2d(32 → 64, k=3, p=1)  → (N, 64,  64,  64)     18,496 params
  ReLU
  MaxPool2d(2)               → (N, 64,  32,  32)
  Flatten                    → (N, 65536)
  Linear(65536 → 256)        → (N, 256)           16,777,472 params
  ReLU
  Linear(256 → 10)           → (N, 10)                 2,570 params   ← logits
```

Convolution output size: $O = \lfloor (I + 2P - K)/S \rfloor + 1$. With `padding=1, k=3, s=1`
the spatial dimensions are unchanged, so height and width are halved purely by the two
`MaxPool2d(2)` layers.

**Why the input is 128 and not 224.** Feeding 224×224 through this same architecture gives a
Flatten of `64×56×56 = 200,704`, so the first `Linear` alone needs **51,380,480** parameters
(**99.96%** of the whole model) — not realistic to train on MPS or CPU. Not a single line of the
architecture was changed; only the input was switched to 128×128, which makes Flatten
`64×32×32 = 65,536`. Part 1 of the notebook still runs the 224 version layer by layer on a
`(4, 3, 224, 224)` dummy input to check it against the architecture table.

| Layer | Params @ 224 | Share |
|---|---:|---:|
| Conv1 | 896 | 0.00% |
| Conv2 | 18,496 | 0.04% |
| Linear1 | 51,380,480 | **99.96%** |
| Linear2 | 2,570 | 0.00% |
| TOTAL | 51,402,442 | |

---

## Training setup

- Loss: `CrossEntropyLoss` (log-softmax is built in, so `forward` returns raw logits — do not
  apply softmax yourself)
- Optimizer: Adam, `lr=1e-3`, `weight_decay=1e-4`
- Scheduler: `CosineAnnealingLR(T_max=12)`
- Augmentation: train uses `RandomResizedCrop(128, scale=(0.5,1.0))` + `RandomHorizontalFlip`;
  val only `Resize(146)` + `CenterCrop(128)`; both normalized with the ImageNet mean / std
- Batch size: 64 train / 128 val

### Training curves

![loss and accuracy curves](assets/02-curves.png)

```
epoch  1/12  train loss 1.9866 acc 33.10%  |  val loss 1.6589 acc 43.18%  (184.1s, lr=9.83e-04)  <- best
epoch  2/12  train loss 1.5364 acc 48.75%  |  val loss 1.4513 acc 51.64%  ( 81.6s, lr=9.33e-04)  <- best
epoch  3/12  train loss 1.3890 acc 54.05%  |  val loss 1.3725 acc 53.78%  ( 48.7s, lr=8.54e-04)  <- best
epoch  4/12  train loss 1.2677 acc 57.56%  |  val loss 1.2010 acc 61.07%  ( 47.3s, lr=7.50e-04)  <- best
epoch  5/12  train loss 1.1721 acc 60.80%  |  val loss 1.2888 acc 58.42%  ( 46.5s, lr=6.29e-04)
epoch  6/12  train loss 1.0842 acc 63.98%  |  val loss 1.1039 acc 64.10%  ( 46.3s, lr=5.00e-04)  <- best
epoch  7/12  train loss 1.0087 acc 67.00%  |  val loss 1.0249 acc 66.55%  ( 43.6s, lr=3.71e-04)  <- best
epoch  8/12  train loss 0.9549 acc 69.10%  |  val loss 0.9932 acc 67.85%  ( 43.0s, lr=2.50e-04)  <- best
epoch  9/12  train loss 0.9058 acc 70.24%  |  val loss 0.9721 acc 68.59%  ( 41.5s, lr=1.46e-04)  <- best
epoch 10/12  train loss 0.8577 acc 72.05%  |  val loss 0.9625 acc 68.82%  ( 45.6s, lr=6.70e-05)  <- best
epoch 11/12  train loss 0.8403 acc 72.90%  |  val loss 0.9364 acc 69.86%  ( 46.1s, lr=1.70e-05)  <- best
epoch 12/12  train loss 0.8233 acc 73.20%  |  val loss 0.9330 acc 70.14%  ( 44.9s, lr=0.00e+00)  <- best
```

Validation accuracy was still climbing at epoch 12 — training stopped because cosine annealing
had driven the learning rate to 0, not because the model turned over into the classic rising
val-loss pattern. But train accuracy (73.20%) is consistently about 3 points above val accuracy
(70.14%), and that gap comes from the 16.8M-parameter `Linear`.

---

## Evaluation

### Per-class accuracy

| Class | Accuracy | Support |
|---|---:|---:|
| tench | 82.69% | 387 |
| English springer | 77.47% | 395 |
| parachute | 76.92% | 390 |
| garbage truck | 76.61% | 389 |
| cassette player | 73.11% | 357 |
| church | 70.90% | 409 |
| French horn | 69.54% | 394 |
| gas pump | 66.35% | 419 |
| golf ball | 60.90% | 399 |
| **chain saw** | **47.41%** | 386 |

### Confusion matrix

![confusion matrix](assets/03-confusion-matrix.png)

The five most frequent confusions:

```
         chain saw → French horn       43 times
         golf ball → parachute         32 times
          gas pump → cassette player   32 times
         chain saw → tench             31 times
         chain saw → English springer  31 times
```

Chain saw (47.41%) drags down the overall score, and it fails in a scattered way — misread as a
French horn, a tench, and an English springer alike. Those photos are typically "a person holding
some long thin metal object, outdoors, cluttered background." Two conv layers only pick up
edge- and color-block-level features, which cannot tell one long thin object from another, so the
model falls back on the background. Golf ball → parachute (a white circle vs. a white canopy) is
the same low-level-feature failure.

### Inference in practice

![predictions on val images](assets/04-predictions.png)

16 randomly sampled validation images, 15 correct (green titles are correct, red are wrong, with
the ground truth in parentheses).

Full softmax distribution for a single image:

![softmax distribution for one image](assets/05-softmax.png)

```
top-3:
  church            93.84%
  gas pump           2.30%
  garbage truck      1.85%
```

---

## What the convolutions actually learned

Left: the weights of all 32 first-layer `3×3×3` filters, drawn directly as RGB thumbnails.
Right: the 32 feature maps produced by `Conv1 + ReLU` on the same church photo.

| Conv1 filters | Conv1 feature maps |
|---|---|
| ![conv1 filters](assets/06-conv1-filters.png) | ![conv1 feature maps](assets/07-feature-maps.png) |

With only two layers, what gets learned is edges, color contrast, and light/dark regions — purely
low-level features. That is exactly why a category like "long thin metal object," which needs
semantic-level features, loses so badly.

This is what the training data looks like after augmentation:

![training batch](assets/01-training-batch.png)

---

## Running it

```bash
git clone https://github.com/lukerspace/ML-CNN-Imagenette-Classifier.git
cd ML-CNN-Imagenette-Classifier
pip install -r requirements.txt
jupyter lab cnn_pytorch.ipynb
```

The dataset is **not in the repo** (129MB). The notebook's `Imagenette(..., download=True)` will
download and extract it into `./data/` on the first run and read it locally from then on.

### Running inference with the trained weights

The checkpoints are 67MB each, so they live in
[Releases](https://github.com/lukerspace/ML-CNN-Imagenette-Classifier/releases) rather than in the repo:

| File | Description |
|---|---|
| `cnn_imagenette.pt` | Final weights, bundled with `classes` / `img_size` / `mean` / `std` / `val_acc` metadata |
| `cnn_imagenette_best.pt` | Best-val-accuracy checkpoint during training (here the last epoch, so identical to the above) |

```bash
mkdir -p checkpoints
gh release download v1.0 -R lukerspace/ML-CNN-Imagenette-Classifier -D checkpoints
```

```python
import torch, torch.nn as nn

class CNN_MLP(nn.Module):                      # same class as notebook section 2.2
    def __init__(self, in_channels=3, img_size=128, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),          nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        side = img_size // 4
        self.flat_dim = 64 * side * side
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, 256), nn.ReLU(), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))


ckpt  = torch.load("checkpoints/cnn_imagenette.pt", map_location="cpu", weights_only=False)
model = CNN_MLP(img_size=ckpt["img_size"], num_classes=len(ckpt["classes"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

`predict_image(path)` in notebook section 2.7 takes any local jpg/png and returns the top-3
classes with probabilities. (It only knows these 10 classes — give it anything else and it will
still pick one.)

---

## Project layout

```
ML-CNN-Imagenette-Classifier/
├── cnn_pytorch.ipynb        # the main content: shape checks → training → evaluation → visualization → save & infer
├── assets/                  # README figures, all extracted from the notebook's outputs
├── requirements.txt
├── data/                    # downloaded automatically, gitignored (129MB)
└── checkpoints/             # downloaded from Releases, gitignored (67MB each)
```

The notebook is in two parts:

| Part | Content |
|---|---|
| Part 1 | Layer-by-layer verification of output shapes and parameter counts against the architecture table, using a `(4, 3, 224, 224)` dummy input |
| Part 2 | The same architecture on Imagenette: data → model → training → evaluation → inference → save / reload |

---

## The bottleneck, and how to push past it

1. **The bottleneck is Flatten, not the convolutions.** 99.96% of the parameters at 224, and
   99.87% at 128, sit in the first `Linear` after `Flatten`. The two conv layers together come to
   19,392 parameters — under 0.1% of the model.
2. **That layer is also the source of the overfitting.** Train accuracy runs above val accuracy,
   and the most effective fix is not more regularization but cutting that layer's parameter count
   first.

The standard ways to improve this, ordered by payoff:

1. Replace the enormous `Flatten` with `AdaptiveAvgPool2d(1)` → one to two orders of magnitude
   fewer parameters, and far less overfitting
2. Add `BatchNorm2d` after each conv → faster, more stable convergence
3. Stack 3–4 conv stages with channels 32 → 64 → 128 → 256 → only then do you get
   semantic-level features (which is what a class like chain saw needs)
4. Add `Dropout`, and switch to SGD + momentum with a longer cosine schedule
5. If you genuinely need high accuracy, use `torchvision.models.resnet18(weights=...)` for
   transfer learning

---

## License

MIT
