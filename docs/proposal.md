# Reproduction Proposal:
## [Dream2Learn: Structured Generative Dreaming for Continual Learning](https://link.springer.com/article/10.1007/s11263-026-02958-1)

I'd like to test the central claim made in the paper: that integrating structured "dreamed classes" improves forward knowledge transfer. Dream2Learn optimizes generated samples toward observed real classes using classifier cross-entropy and stops before collapse or overspecialization; it does not directly maximize uncertainty.

The authors' approach is inspired by how humans, while dreaming, generate imagined experiences that enhance learning and memory retention outcomes. The authors suggest that by integrating these dreamed experiences into the continual learning process, models can organize their latent feature space more effectively, leading to improved generalization without catastrophic forgetting.

However, the authors' approach relies on a pre-trained diffusion model to generate the dreamed classes. This presents the risk of target leakage, where the pre-trained model inadvertently introduces information about the target classes into the dreamed classes, potentially inflating the model's performance in a way that does not reflect true forward knowledge transfer. The authors acknowledge this risk but argue that it's impact is minimal and does not significantly undermine the validity of their results.

Thus, in my reproduction of their work, I plan to evaluate the validity of their core claim by eliminating the potential influence of target leakage from the pre-trained diffusion model and assessing whether the Dream2Learn algorithim still contributes to improved forward knowledge transfer.

To do this, the implementation uses a conditional variational autoencoder with a label-free encoder and classifier, and a decoder conditioned only on observed real labels. Class-directed latent transformations generate auxiliary classes without any external pretrained component. Only current-task training data and bounded real replay may guide learning or dreaming. Decoded images are diagnostics, not training examples.

This is a leakage-controlled latent adaptation, not an exact numerical reproduction or a claim that latent and image-mediated training are equivalent. Matched CVAE/replay controls, a measured random-initialization transfer reference, and multiple seeds are needed to assess the benefit. The protocol, limitations, and runnable notebook are documented in [README.md](README.md).

Ipfi Lidzhade
1037819
