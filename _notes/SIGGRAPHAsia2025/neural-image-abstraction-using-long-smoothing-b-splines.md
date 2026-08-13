---
title: "Neural Image Abstraction Using Long Smoothing B-Splines"
authors:
  - "Daniel Berio"
  - "Michael Stroh"
  - "Sylvain Calinon"
  - "Frederic Fol Leymarie"
  - "Oliver Deussen"
  - "Ariel Shamir"
category: "Image & Video"
track: "Journal"
source: "arxiv"
institution:
  - "University of London"
  - "University of Konstanz"
  - "Idiap Research Institute"
  - "Reichman University"
tags:
  - "Differentiable Vector Graphics"
  - "B Splines"
  - "Image Abstraction"
  - "Vectorization"
  - "Stroke Based Rendering"
  - "Diffusion"
  - "CLIP"
  - "Text Stylization"
links:
  paper: "https://doi.org/10.1145/3763345"
  project: "https://github.com/colormotor/calligraph"
  code: "https://github.com/colormotor/calligraph"
---

## Overview

This work targets the generation of long, smooth, expressive curves for stylized vector graphics inside a differentiable vector graphics (DiffVG) rasterization pipeline. Most existing DiffVG-based methods directly optimize piecewise Bézier curves, and even with added smoothing penalties they cannot guarantee continuity across segments, which limits their ability to represent long and flowing strokes. The key idea is to instead parameterize curves as uniform smoothing B-splines, which have inherent high-order continuity, and convert them exactly to piecewise cubic Béziers through a linear mapping so they can be rendered and optimized with the standard DiffVG method. Because the conversion is a matrix multiplication and rendering is differentiable, image-space gradients back-propagate to the B-spline key-points.

The method treats stroke width as a third curve dimension, so each control point carries an independent stroke radius, enabling smooth width variation resembling physical brush strokes; allowing the width to vanish provides an effective way to control the effective number of visible strokes. The approach supports both open (clamped) and closed (periodic) curves, and is demonstrated on four applications: stylized space-filling / area fills, single-stroke image abstraction, area-based image abstraction with color quantization, and text stylization / calligram generation.

## Method

A B-spline of degree $$p$$ and order $$k = p + 1$$ is expressed as a linear combination of shifted basis functions:

$$\boldsymbol{x}(u) = \sum_{i=0}^{n-1} \boldsymbol{c}_i \, N_k(u - t_i)$$

over $$n$$ control points and a fixed non-decreasing knot sequence of $$m = n + k$$ uniformly spaced integer knots. Uniform ("cardinal") B-splines are chosen for their simplicity, high-order continuity, and analytic derivatives and integrals. Rather than exposing raw control points, a user specifies a spline through $$M$$ key-points $$\boldsymbol{Q} = \boldsymbol{q}_1, \dots, \boldsymbol{q}_M$$, which are optimized directly. Clamped (open) splines pad the first and last key-point $$k-1$$ times, producing curves that begin and end at rest; periodic (closed) splines append the first $$k-1$$ key-points. Following prior calligraphy work, repeated control points (rather than repeated knots) are used to maintain strict uniformity while enabling adaptive smoothing of corner-like features. Repeating a key-point initially lowers continuity by one degree, giving extra degrees of freedom that the optimizer can adaptively smooth.

### Smoothing cost

Smoothness is imposed through the squared magnitude of positional derivatives, a criterion standard in curve fairing and in modeling human arm motion:

$$\mathcal{L}^d_{\text{smooth}} = \frac{1}{T} \int_{t_{k-1}}^{t_{m-k}} \| \boldsymbol{x}^{(d)}(u) \|^2 \, du = \frac{1}{T} \, \boldsymbol{c}^\top \bar{\boldsymbol{G}} \, \boldsymbol{c}$$

where $$T = t_{m-k} - t_{k-1}$$ and $$\boldsymbol{c}$$ concatenates all control points. The integral is computed exactly by building $$\bar{\boldsymbol{G}}$$ as a block Gram matrix from inner products of basis-function derivatives; a finite-difference approximation instead yields the penalized-spline (P-spline) formulation. Because the matrix is precomputed per stroke, both variants have similar runtime. Most examples use quintic splines with a smoothing cost on the third positional derivative (jerk), motivated by the "minimum jerk" principle from motor control and its role as an approximant for curvature variation in fairing. The formulation generalizes to other spline degrees and smoothing orders (e.g. snap, the fourth derivative).

### Conversion to Bézier and rendering

B-splines are converted exactly to piecewise Bézier curves of the same degree via a block transformation matrix $$\boldsymbol{S}$$ (matrix multiplication on the flattened control points). Since higher-degree curves needed for jerk/snap smoothing are not natively renderable in DiffVG, the Bézier curves are degree-reduced to cubic via a second block matrix $$\bar{\boldsymbol{R}}$$, so the final cubic Bézier control points are computed as $$\bar{\boldsymbol{R}} \bar{\boldsymbol{S}} \boldsymbol{c}$$. This reduction introduces negligible geometric error (reported as less than 0.3% of the curve's bounding-box diagonal). The resulting control points live in $$\mathbb{R}^3$$, where the third dimension is stroke radius; control points, stroke colors, and fill colors are all differentiable parameters, and rendering produces an image differentiable with respect to them.

### Combined objective

All applications use a combined cost splitting image-space and geometric terms:

$$\mathcal{L} = \mathcal{L}_I + \mathcal{L}_G$$

where $$\mathcal{L}_I$$ relies on differentiable rasterization (raster-based objectives) and $$\mathcal{L}_G$$ leverages B-spline properties for smoothing, stylization, and constraints while preserving continuity. Optimization uses Adam with cosine-annealed learning rates, typically 300 steps taking roughly 30–60 seconds on a single RTX 3060; diffusion-guided runs cost about 0.6–1.0 s per step (up to about 6 minutes).

## Applications

### Area fillings and pattern generation

A baseline that fills solid regions with stylized patterns. Initialization uses weighted Voronoi stippling on a bitmap or saliency map, with points connected into an open or looping TSP path ("TSP art"). The image coverage term is a multiscale MSE loss between target and rendered image, where lower scales encourage alignment with broad intensity regions and faster convergence and higher scales promote accurate silhouette reconstruction; reducing target opacity directly reduces stroke density. An optional bounding-box loss keeps key-points inside a target region using Softplus or ReLU penalties:

$$\mathcal{L}_{\text{box}} = \sum_i \boldsymbol{1}^\top \left( \varphi(\boldsymbol{b}_{\min} - \boldsymbol{p}_i) + \varphi(\boldsymbol{p}_i - \boldsymbol{b}_{\max}) \right)$$

A semantic stylization term based on a patch-wise directional CLIP loss (using the robustified CLIPAG ViT-B/32) enables stylization from a text prompt or an example image.

### Single-stroke image abstraction

Image-conditioned stylization is achieved by combining Score Distillation Sampling with ControlNet (Canny edges) and IP-Adapter conditioning; ControlNet preserves structure while IP-Adapter aligns with global appearance. A generic prompt such as "A black and white drawing" suffices. The SDS-style gradient has the form:

$$\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_t \left[ \omega(t) \left( \boldsymbol{\epsilon}_\phi(x_t, t, y) - \boldsymbol{\epsilon} \right) \frac{\partial g(\theta)}{\partial \theta} \right]$$

The Interval Score Matching (ISM) variant with time-step annealing improves convergence and allows standard classifier-free guidance of 7.5. Because higher denoising time steps yield coarser features while single strokes lack degrees of freedom for fine detail, the minimum time step is limited (around 500), and strokes are initialized with multiplicity greater than one (e.g. 3 for quintics) for smoother results. Letting a single stroke reach zero width provides an automatic way to determine the effective number of strokes.

### Area-based image abstraction

Smooth closed areas are generated for designs with overlapping regions and limited color palettes. The image term uses a CLIP-driven geometric cost (layers 2 and 3 of CLIPAG) with an $$L_1$$ norm:

$$\mathcal{L}_{\text{CLIP}} = \sum_l \left\| \text{CLIP}_l(\hat{\mathbf{I}}) - \text{CLIP}_l(\mathbf{I}_\theta) \right\|_1$$

A tangent-point repulsion energy penalizes self-intersections and overlaps within each area. Discrete color assignment from a fixed palette $$\boldsymbol{V} \in \mathbb{R}^{K \times 3}$$ is made differentiable with the Gumbel-Softmax trick, using soft assignments

$$\boldsymbol{a}_i = \text{softmax}\left( \frac{\boldsymbol{\ell}_i + \boldsymbol{g}_i}{\tau} \right), \quad \boldsymbol{g}_i \sim \text{Gumbel}(0, \beta)^K$$

with an annealed temperature $$\tau$$, a noise scale $$\beta$$ around 0.15, and a regularization term encouraging balanced palette usage. Hard colors for visualization come from an argmax over the logits. Areas are initialized from Voronoi sampling on a saliency map and sorted by increasing saliency.

### Text stylization

Smooth text abstractions and calligrams are fit into a target silhouette. Starting from a target bitmap and an initial text layout, glyph outlines are uniformly sampled into key-point sequences, then deformed under a loss balancing silhouette coverage, outline smoothness, and point repulsion. To prevent smoothing from destroying readability, a novel legibility loss computes an $$L_1$$ distance between the last-layer [CLS] token features (from the TrOCR vision encoder) of the deformed rendering and the original layout. Automatic glyph placement optimizes a per-glyph similarity transform combining a coverage term, an overlap cost given by $$\sum \text{ReLU}(v - 0.5)$$ over pixel intensities, and an alignment cost $$\sum \| \theta \|$$ penalizing turning angles between consecutive glyph centers to maintain ordering. Silhouettes can be produced automatically from text-to-image prompts.

## Discussion

The central takeaway is that reparameterizing curves as smoothing B-splines within a DiffVG pipeline enforces high-order continuity by construction and enables analytic smoothing losses, producing more regular geometry than direct Bézier or lower-continuity parameterizations. Combining smoothing with optimizable stroke width lets the fidelity-vs-simplicity trade-off be controlled parametrically, with the number of strokes emerging from optimization rather than being fixed in advance, giving a simpler solution than prior stroke-based abstraction methods. The Voronoi-based initialization is used for consistency but is not required, and different initializations serve as a design parameter. Because quintic splines with jerk smoothing produce smooth speed and acceleration profiles (slowing where curvature is higher, mirroring human hand motion), the resulting paths can be resampled and tracked by a robot, with the variable width mapping to brush pressure for physical reproduction. Working code and examples are available in the authors' repository.
