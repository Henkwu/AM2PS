# Paper-to-code implementation notes

This repository is a **reference reimplementation from the manuscript**, not the authors' original training code. The manuscript gives the high-level architecture and several hyperparameters, but omits some details required for exact numerical reproduction.

## Directly specified by the manuscript

- Backbone: pretrained ResNet50.
- Vision-language model: CLIP ViT-B/32.
- CLIP visual and text encoders: treated as frozen in the method description.
- Data augmentation: random flipping, rotation, and color jittering.
- Scalable adapter: 2 layers with linear transformations; ReLU is shown/described.
- Multi-scale fusion: 3 scales selected as best; subsequent scales halve the feature dimension.
- Transformer refinement: 3 layers selected as best.
- Fusion weights: lambda1 = 1, lambda2 = 1.
- Multi-scale weights: alpha = 1/3.
- Optimizer: Adam.
- Initial learning rate: 2e-4.
- Weight decay: 1e-4.
- Epochs: 60.
- Scheduler: cosine annealing.
- Prompts: `a photo of a {class}`, `a clinical photo of a {class}`, `a chest X-ray photo of a {class}`.

## Details not specified precisely in the manuscript

The manuscript does not uniquely specify batch size, random seed, input resolution, rotation magnitude, color-jitter magnitude, adapter bottleneck width, attention-head count, precise projection implementation, or the validation construction for ChestXRay-Covid19. Defaults in the YAML files are therefore explicit implementation choices rather than claimed original settings.

## Two equation-level ambiguities handled explicitly

1. **Inter-modal probability (Eq. 10--11).** The manuscript normalizes prompt similarities over the K prompts within each class and then averages those normalized values. Taken literally, the mean is exactly `1/K` for every class, making the branch non-discriminative. The default implementation therefore uses a standard prompt ensemble: softmax over classes for each prompt, followed by averaging over prompts. Set `model.inter_modal_mode: paper_literal` only to inspect the literal equation behavior.

2. **Probability fusion (Eq. 22).** The manuscript uses `lambda1 * P_inter + lambda2 * P_intra` inside a log with lambda1=lambda2=1, without an explicit normalization term. The default code normalizes the weighted sum to a valid probability distribution. Set `model.normalize_probability_fusion: false` only for equation-level inspection.

These choices are documented so that future revisions of the manuscript and code can be kept consistent.
