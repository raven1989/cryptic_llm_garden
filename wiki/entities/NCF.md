---
tags: ["entity", "architecture", "recommendation", "collaborative-filtering", "neural-networks", "implicit-feedback"]
aliases: ["NCF", "Neural Collaborative Filtering", "NeuMF", "GMF"]
date: 2026-08-17
sources: ["[[wiki/research/NCF Summary.md]]"]
---

# NCF

**NCF (Neural Collaborative Filtering)** is a general recommendation framework published in 2017 (WWW) by Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. It replaces matrix factorization's **fixed inner product** with a **neural network that learns the user–item interaction function from data**, targeting **implicit feedback**. MF is recovered as a special case; non-linearity is added via an MLP; and the two are fused in **NeuMF**. It is the canonical ancestor of the embedding + MLP / two-tower paradigm in modern recommenders.

![NCF Framework](../media/NCF_general_framework.png)

*The NCF framework: one-hot ID input → embedding lookup → neural CF layers → sigmoid prediction.*

## Core Architectural Pillars

### 1. Input & Embedding (ID → latent vector)
Inputs are **one-hot ID encodings** — $\mathbf{v}^U_u \in \{0,1\}^M$, $\mathbf{v}^I_i \in \{0,1\}^N$ — not interaction histories. The fully connected embedding layer is a **row lookup** into latent factor matrices:

$$\mathbf{p}_u = \mathbf{P}^T \mathbf{v}^U_u \in \mathbb{R}^K, \qquad \mathbf{q}_i = \mathbf{Q}^T \mathbf{v}^I_i \in \mathbb{R}^K$$

These embeddings *are* the classical MF latent vectors, learned end-to-end. The framework is generic: content/context features can replace IDs to handle cold-start. (Contrast with [[DSSM]], whose towers consume rich text features; NCF's towers collapse to embedding lookups.)

### 2. Neural CF Layers (the learned interaction function)
The interaction function $f$ is a composition of learned layers:

$$\hat{y}_{ui} = \phi_{out}\big(\phi_X(\dots \phi_1(\mathbf{p}_u, \mathbf{q}_i)\dots)\big)$$

The dimension of the last hidden layer $X$ is the model's capacity knob (**predictive factors**). Three instantiations differ only here:

| Model | First CF layer $\phi_1$ | Rest | Output |
| :--- | :--- | :--- | :--- |
| **GMF** | element-wise product $\mathbf{p}_u \odot \mathbf{q}_i$ | — (direct projection) | $\sigma(\mathbf{h}^T(\mathbf{p}_u \odot \mathbf{q}_i))$ |
| **MLP** | concatenation $[\mathbf{p}_u; \mathbf{q}_i]$ | ReLU hidden layers, tower-halved ($32{\to}16{\to}8$) | $\sigma(\mathbf{h}^T \mathbf{z}_L)$ |
| **NeuMF** | separate embeddings per branch | GMF product ∥ MLP tower, concatenated | $\sigma(\mathbf{h}^T[\phi^{GMF}; \phi^{MLP}])$ |

- **GMF** with identity $a_{out}$ and $\mathbf{h}{=}\mathbf{1}$ *exactly recovers* matrix factorization; learning $\mathbf{h}$ + sigmoid generalizes it.
- **MLP** needs hidden layers to create interaction — MLP-0 (bare concatenation) performs at ItemPop level.
- **NeuMF** deliberately uses **separate embeddings** for the two branches (sharing would force equal embedding sizes).

### 3. Sigmoid Output + Log Loss (binary classification)
The output activation is a sigmoid, so $\hat{y}_{ui} \in [0,1]$ is the probability item $i$ is relevant to $u$. Training minimizes **binary cross-entropy (log loss)**:

$$L = -\sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log(1 - \hat{y}_{ui})$$

This recasts recommendation from implicit feedback as binary classification — a better fit than squared loss (which assumes Gaussian targets) for binary data.

### 4. Negative Sampling
Negatives $\mathcal{Y}^-$ are **uniformly sampled** from unobserved entries each iteration. Pointwise log loss allows a flexible sampling ratio (unlike pairwise BPR, fixed at one negative per positive). Optimal ratio ≈ **3–6**; too aggressive (>7) hurts.

---

## Training Details

- **GMF / MLP from scratch:** Gaussian init ($\mu{=}0,\sigma{=}0.01$), mini-batch **Adam** (faster convergence than vanilla SGD).
- **NeuMF:** **pre-train** GMF and MLP to convergence, use their parameters as initialization, fuse output weights as $\mathbf{h} \leftarrow [\alpha \mathbf{h}^{GMF}; (1{-}\alpha)\mathbf{h}^{MLP}]$ with $\alpha{=}0.5$, then fine-tune with **vanilla SGD** (Adam needs momentum state not carried over from pre-training).

## Key Properties

| Property | Detail |
| :--- | :--- |
| **Input** | One-hot user/item ID (generic: content/context features possible) |
| **Embedding** | Row lookup into $\mathbf{P} \in \mathbb{R}^{M\times K}$, $\mathbf{Q} \in \mathbb{R}^{N\times K}$ |
| **Interaction** | Learned (GMF product / MLP tower / NeuMF fusion) — not fixed inner product |
| **Output activation** | Sigmoid → probability of relevance |
| **Loss** | Binary cross-entropy (log loss) |
| **Negative sampling** | Uniform, ratio ~3–6 per positive |
| **Optimizer** | Adam (GMF/MLP); pre-train + vanilla SGD (NeuMF) |
| **Capacity knob** | Predictive factors = last hidden layer dim $\{8,16,32,64\}$ |

---

## Experimental Results

Leave-one-out evaluation, test item ranked among 100 sampled negatives (HR@10 / NDCG@10):

- **NeuMF** beats eALS by **4.5%** and BPR by **4.9%** relative on average; on Pinterest, NeuMF with 8 factors beats eALS/BPR with 64. All gains significant ($p<0.01$).
- **GMF > BPR** (same MF class, log loss vs. pairwise) — validates the log-loss treatment.
- **Depth:** MovieLens HR@10 (8 factors) 0.452 → 0.628 → 0.655 → 0.671 → 0.678 for MLP-0→4; linear stacking does *not* help — gains come from non-linearity.
- **Pre-training** NeuMF helps (MovieLens 64 factors: HR 0.730 vs. 0.705).

---

## Related Wiki Pages
* [[NCF Summary]]: Complete, section-by-section research summary of the paper — including the MF inner-product limitation, the implicit-feedback loss strategies (WMF/BPR/eALS vs. log loss), and negative sampling.
* [[DSSM]]: Two-tower semantic matching with rich input features — same shape, different inputs.
* [[DCN]]: Deep & Cross Network — explicit bounded-degree feature crossing for CTR, a complementary 2017 architecture.
* [[EASE]]: An embarrassingly shallow linear item–item CF alternative with a closed-form solution.
