# Rak-Embedding-Model

RakEmbeddingModel is a self-supervised image embedding model built completely from scratch using PyTorch.  
It implements contrastive representation learning using a CNN encoder and the InfoNCE loss objective.

---

## Cluster Formation During Training

The animation below shows how embeddings evolve across training epochs.  
Each point represents an image embedding projected to 2D using UMAP.  
Clusters tighten and separate as training progresses.

<!-- Replace with your actual GIF path -->
![Cluster Evolution](path_to_cluster_evolution.gif)

---

## Model Architecture

RakEmbeddingModel follows a SimCLR-style architecture composed of:

- A Convolutional Neural Network (CNN) encoder  
- A Projection Head (MLP)  
- InfoNCE contrastive loss  

The forward path is:
Image → CNN Encoder → Embedding (h) → Projection Head → Projection (z)


- The encoder produces the final embedding `h`.
- The projection head produces `z`, used only during contrastive training.
- Only the encoder output is used for evaluation and downstream tasks.

<!-- Replace with your actual architecture image path -->
![RakEmbeddingModel Architecture](path_to_architecture_image.png)

---

## Why CNN for Embedding Learning

Convolutional Neural Networks are used because they:

- Capture spatial hierarchies in images
- Detect edges, curves, and structural patterns
- Preserve locality through convolution
- Provide translation invariance
- Scale efficiently to image data

The CNN encoder learns structured representations that organize semantically similar inputs into nearby regions of embedding space.

---

## Why Use InfoNCE Loss

RakEmbeddingModel uses the InfoNCE (Normalized Temperature-Scaled Cross Entropy) objective.

Training procedure:

- Two augmented views of the same image are generated.
- They form a positive pair.
- All other samples in the batch act as negatives.
- The model learns to maximize similarity between positives and minimize similarity with negatives.

Mathematically:

\[
\mathcal{L} = - \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}
{\sum_{k=1}^{2B} \exp(\text{sim}(z_i, z_k)/\tau)}
\]

Where:
- `sim` is cosine similarity
- `τ` is temperature
- `B` is batch size

InfoNCE is used because:

- It leverages the entire batch as negative samples
- It provides strong optimization signals
- It produces highly structured embedding spaces
- It enables self-supervised learning without labels

---

## Why Use a Projection Head

The projection head maps encoder embeddings into a space optimized for contrastive learning.

This separation ensures:

- The encoder learns general, transferable features
- The contrastive objective operates in a flexible training space
- Final embeddings remain useful for downstream tasks

During inference, only the encoder output is used.

---

## Why UMAP for Visualization

Embeddings are high-dimensional (e.g., 128D).  
To visualize cluster formation, dimensionality reduction is required.

UMAP is used because it:

- Preserves local neighborhood structure
- Maintains cluster topology
- Produces stable 2D projections
- Scales efficiently

UMAP is fitted across embeddings from all epochs to ensure consistent coordinates in the animation.

---

## Summary

RakEmbeddingModel demonstrates:

- Self-supervised representation learning from scratch
- Contrastive training with InfoNCE
- CNN-based embedding extraction
- Projection head stabilization
- Quantitative embedding evaluation
- Temporal visualization of cluster evolution

The project focuses on understanding and implementing modern embedding learning mechanisms at a foundational level.
