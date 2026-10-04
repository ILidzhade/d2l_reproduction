# Reproduction Proposal:
## [Dream2Learn: Structured Generative Dreaming for Continual Learning](https://link.springer.com/article/10.1007/s11263-026-02958-1)

I'd like to test the central claim made in the paper, in which the authors improved foward knowledge transfer by integrating "dreamed classes", that maximize uncertainty accross known classes, into the continual learning process.

The authors' approach is inspired by how humans, while dreaming, generate imagined experiences that enhance learning and memory retention outcomes. The authors suggest that by integrating these dreamed experiences into the continual learning process, models can organize their latent feature space more effectively, leading to improved generalization without catastrophic forgetting.

However, the authors' approach relies on a pre-trained diffusion model to generate the dreamed classes. This presents the risk of target leakage, where the pre-trained model inadvertently introduces information about the target classes into the dreamed classes, potentially inflating the model's performance in a way that does not reflect true forward knowledge transfer. The authors acknowledge this risk but argue that it's impact is minimal and does not significantly undermine the validity of their results.

Thus, in my reproduction of their work, I plan to evaluate the validity of their core claim by eliminating the potential influence of target leakage from the pre-trained diffusion model and assessing whether the Dream2Learn algorithim still contributes to improved forward knowledge transfer.

To do this, I plan to create a bottleneck classifier that generates dreamed classes directly within it's own latent space. This approach will allow me to isolate the effect of the Dream2Learn algorithim on forward knowledge transfer without the confounding influence of a pre-trained diffusion model.

Ipfi Lidzhade
1037819
