# Week 4 — PyTorch / ExecuTorch

Train a model, then ship it to run on-device the way openpilot does: tensors & autograd,
training loops, CNNs, exporting, ExecuTorch lowering, and quantization.

> Needs `torch`. CPU-only is fine: `pip install torch` (or the CPU index if you prefer).
> ExecuTorch (Day 27) uses `torch.export`; the full runtime installs separately — see its lesson.

| Day | Topic | Status |
|-----|-------|--------|
| 22 | [Tensors & autograd](day22_tensors_autograd) | ✅ built |
| 23 | [`nn.Module` & the training loop](day23_nn_module) | ✅ built |
| 24 | [CNNs & computer vision](day24_cnns) | ✅ built |
| 25 | [Datasets, DataLoaders & augmentation](day25_dataloaders) | ✅ built |
| 26 | [Exporting models (TorchScript / ONNX / `torch.export`)](day26_exporting) | ✅ built |
| 27 | [ExecuTorch: on-device inference](day27_executorch) | ✅ built |
| 28 | [Quantization & making models fast](day28_quantization) | ✅ built |

The week is a pipeline: **train (22–24) → feed data (25) → export (26) → lower on-device (27) →
quantize (28)** — the same path openpilot's driving model takes from training to the car.

See [SYLLABUS.md](../../SYLLABUS.md) for the full plan.
