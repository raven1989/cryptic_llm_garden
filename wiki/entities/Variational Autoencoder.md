---
tags:
  - generative-models
  - architecture
  - math
  - probability
date: 2026-06-17
sources: ["[[raw/recommendation/一文详解codebook技术史.md]]"]
---

# Variational Autoencoder (VAE) Deep Dive

This page provides a rigorous, mathematically robust treatment of the **Variational Autoencoder (VAE)**. It outlines its underlying probability theory, includes complete proofs of its core components (Jensen's inequality, KL divergence non-negativity, and Gaussian closed-form KL loss), and details why its objective function naturally maps to standard regression/classification error terms.

> [!WARNING] **Notation Heads Up (Non-Standard $p$ and $q$ definitions)**
> In standard VAE literature (e.g., Kingma & Welling, 2013), the **encoder / posterior** is denoted as $q_\phi(z|x)$, the **decoder / likelihood** is denoted as $p_\theta(x|z)$, and the **latent prior** is denoted as $p(z)$.
> 
> Throughout this document (inherited from the raw source material `raw/Recommendation/一文详解codebook技术史.md`), the notation letters are **switched**:
> - **Encoder / Posterior:** $p_\phi(z \vert x)$
> - **Decoder / Likelihood:** $q_\theta(x \vert z)$
> - **Latent Prior:** $q(z)$
> - **Joint Empirical Process:** $p(x, z) = p_{\text{data}}(x) p_\phi(z \vert x)$
> - **Joint Generative Model:** $q(x, z) = q(z) q_\theta(x \vert z)$
> - **ELBO Formulation:** $\text{ELBO}(x) = \mathbb{E}_{z \sim p_\phi(z \vert x)}[\log q_\theta(x \vert z)] - KL\big(p_\phi(z \vert x) \parallel q(z)\big)$
> 
> Though the letter assignments are inverted relative to standard deep learning conventions, the mathematical proofs and PyTorch implementation are **internally fully consistent and correct**.

---

## 1. Fundamental Mathematical Tools

### A. Geometric Proof of Jensen's Inequality (Concave Version)

#### **Theorem**
If $\varphi(x)$ is a concave function ($\cap$), then the expectation of the function is less than or equal to the function of the expectation:
$$\mathbb{E}[\varphi(X)] \le \varphi(\mathbb{E}[X])$$

#### **Proof**
1. **Define the Point of Tangency:**
   Let $X$ be a random variable, and define $\mu = \mathbb{E}[X]$ to be its expected value.
2. **Construct the Tangent (Supporting) Line:**
   The equation of the tangent line $L(x)$ touching the curve at the point $(\mu, \varphi(\mu))$ with slope $a = \varphi'(\mu)$ is:
   $$L(x) = a(x - \mu) + \varphi(\mu)$$
3. **Apply Concavity:**
   By definition, a concave function lies strictly on or below its tangent/supporting lines:
   $$\varphi(x) \le a(x - \mu) + \varphi(\mu)$$
4. **Take the Expectation on Both Sides:**
   Substituting our random variable $X$ and taking expectations:
   $$\mathbb{E}[\varphi(X)] \le \mathbb{E}[a(X - \mu) + \varphi(\mu)]$$
5. **Expand via Linearity of Expectation:**
   Since $a, \mu, \text{and } \varphi(\mu)$ are constants under $\mathbb{E}$:
   $$\mathbb{E}[a(X - \mu) + \varphi(\mu)] = a(\mathbb{E}[X] - \mu) + \varphi(\mu)$$
6. **Eliminate the Linear Term:**
   Since we defined $\mu = \mathbb{E}[X]$, we substitute it back:
   $$a(\mu - \mu) + \varphi(\mathbb{E}[X]) = a(0) + \varphi(\mathbb{E}[X]) = \varphi(\mathbb{E}[X])$$
7. **Conclusion:**
   Substituting this back into the inequality completes the proof:
   $$\mathbb{E}[\varphi(X)] \le \varphi(\mathbb{E}[X]) \quad \blacksquare$$

---

### B. Proof of the Non-Negativity of KL Divergence ($KL \ge 0$)

The Kullback-Leibler (KL) divergence measures the difference between two probability distributions $p_1(x)$ and $p_2(x)$.

#### **Theorem**
$$KL(p_1(x) \parallel p_2(x)) \ge 0$$

#### **Proof**
1. **Define KL Divergence:**
   $$KL(p_1(x) \parallel p_2(x)) = \int p_1(x) \log \frac{p_1(x)}{p_2(x)} dx$$
2. **Rewrite as a Negative Expectation:**
   Using the logarithm identity $\log(A/B) = -\log(B/A)$:
   $$-KL(p_1(x) \parallel p_2(x)) = \int p_1(x) \log \frac{p_2(x)}{p_1(x)} dx = \mathbb{E}_{X \sim p_1}\left[ \log \left( \frac{p_2(X)}{p_1(X)} \right) \right]$$
3. **Determine the Expectation of the Ratio:**
   Let $T = \frac{p_2(X)}{p_1(X)}$. Let's compute its expected value under $X \sim p_1(X)$:
   $$\mathbb{E}_{X \sim p_1}[T] = \int p_1(x) \left( \frac{p_2(x)}{p_1(x)} \right) dx = \int p_2(x) dx = 1$$
   *(Since $p_2(x)$ is a valid probability density function, its total integral is $1$).*
4. **Apply Jensen's Inequality for the Concave Function $\varphi(t) = \log(t)$:**
   $$\mathbb{E}_{X \sim p_1}[\log(T)] \le \log(\mathbb{E}_{X \sim p_1}[T])$$
5. **Substitute Values:**
   * Left side: $-KL(p_1(x) \parallel p_2(x))$
   * Right side: $\log(1) = 0$
   $$-KL(p_1(x) \parallel p_2(x)) \le 0$$
6. **Conclusion:**
   Multiplying by $-1$ flips the inequality sign:
   $$KL(p_1(x) \parallel p_2(x)) \ge 0 \quad \blacksquare$$
   *(Equality holds if and only if $T = 1$ with probability 1, meaning $p_1(x) = p_2(x)$ almost everywhere).*

---

## 2. Theoretical Framework (Joint Probability Perspective)

Rather than modeling the marginal log-likelihood $\log p(x)$ point-wise, VAEs can be formulated by optimizing the closeness of the joint distribution of the empirical data and encoder process $p(x, z)$ to our modeled prior and decoder process $q(x, z)$.

We define the total KL divergence between these joint distributions, which must be non-negative:
$$KL\big(p(x, z) \parallel q(x, z)\big) = \iint p(x, z) \log \frac{p(x, z)}{q(x, z)} dz dx \ge 0$$

Applying the **Bayesian substitutions**:
*   **Empirical joint process:** $p(x, z) = p_{\text{data}}(x) p_\phi(z|x)$
*   **Generative joint model:** $q(x, z) = q(z) q_\theta(x|z)$

We rewrite the double integral:
$$\iint p_{\text{data}}(x) p_\phi(z|x) \log \frac{p_{\text{data}}(x) p_\phi(z|x)}{q(z) q_\theta(x|z)} dz dx \ge 0$$

By pulling out the expectation over the data $\mathbb{E}_{x \sim p_{\text{data}}(x)}$ and splitting the logarithm, the expression simplifies directly to:
$$\mathbb{E}_{x \sim p_{\text{data}}(x)}[\log p_{\text{data}}(x)] - \mathbb{E}_{x \sim p_{\text{data}}(x)}[\text{ELBO}(x)] \ge 0$$
$$\mathbb{E}_{x \sim p_{\text{data}}(x)}[\log p_{\text{data}}(x)] \ge \mathbb{E}_{x \sim p_{\text{data}}(x)}[\text{ELBO}(x)]$$

Where the **Evidence Lower Bound (ELBO)** for a single data point is defined as:
$$\text{ELBO}(x) = \mathbb{E}_{z \sim p_\phi(z|x)}[\log q_\theta(x|z)] - KL\big(p_\phi(z|x) \parallel q(z)\big)$$

To optimize the model, we maximize the expected ELBO, which is equivalent to minimizing the total loss function:
$$\text{Total Loss} = \mathbb{E}_{x \sim p_{\text{data}}(x)}\Big[ -\mathbb{E}_{z \sim p_\phi(z|x)}[\log q_\theta(x|z)] + KL\big(p_\phi(z|x) \parallel q(z)\big) \Big]$$

---

## 3. Translation to Practical Loss Functions

### A. Term 1: Reconstruction Loss
To minimize $-\mathbb{E}_{z \sim p_\phi(z|x)}[\log q_\theta(x|z)]$, we assume a distribution for $q_\theta(x|z)$ based on our data types:

1. **Continuous Data (Mean Squared Error - MSE):**
   Assume the reconstructed image follows a Gaussian distribution centered at the decoder's prediction $\hat{x} = \text{Decoder}(z)$:
   $$q_\theta(x|z) = \mathcal{N}(x; \hat{x}, \sigma^2 \mathbf{I}) \implies -\log q_\theta(x|z) = \frac{\|x - \hat{x}\|_2^2}{2\sigma^2} + C$$
   Thus, minimizing the negative log-likelihood is mathematically identical to minimizing the **MSE**.
   
2. **Normalized/Binary Data (Binary Cross-Entropy - BCE):**
   Assume each dimension $x_i \in [0, 1]$ follows a Bernoulli distribution where the probability is predicted as $\hat{x}_i = \text{Decoder}(z)_i$:
   $$q_\theta(x|z) = \prod_i \hat{x}_i^{x_i} (1 - \hat{x}_i)^{1-x_i} \implies -\log q_\theta(x|z) = -\sum_i \left[ x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i) \right]$$
   Thus, minimizing the negative log-likelihood is mathematically identical to minimizing **BCE**.

---

### B. Term 2: Closed-form Gaussian KL Loss
We assume the latent prior $q(z) = \mathcal{N}(0, \mathbf{I})$ and the encoder posterior $p_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 \mathbf{I})$.

For a single dimension, let $p(z) = \mathcal{N}(\mu, \sigma^2)$ and $q(z) = \mathcal{N}(0, 1)$. We expand:
$$KL(p \parallel q) = \mathbb{E}_p[\log p(z)] - \mathbb{E}_p[\log q(z)]$$

Taking the logarithm of the probability densities:
$$\log p(z) = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}$$
$$\log q(z) = -\frac{1}{2}\log(2\pi) - \frac{z^2}{2}$$

Now we take the expected value $\mathbb{E}_p[\cdot]$ of both logs under $z \sim \mathcal{N}(\mu, \sigma^2)$:

1. **First Term:**
   $$\mathbb{E}_p[\log p(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{\mathbb{E}_p[(z-\mu)^2]}{2\sigma^2}$$
   Since $\mathbb{E}_p[(z-\mu)^2]$ is the definition of variance, it equals $\sigma^2$:
   $$\mathbb{E}_p[\log p(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{1}{2}$$

2. **Second Term:**
   $$\mathbb{E}_p[\log q(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\mathbb{E}_p[z^2]$$
   Using the algebraic identity $\mathbb{E}[z^2] = \text{Var}(z) + (\mathbb{E}[z])^2 = \sigma^2 + \mu^2$:
   $$\mathbb{E}_p[\log q(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}(\sigma^2 + \mu^2)$$

3. **Subtract the expectations:**
   $$KL(p \parallel q) = \left[ -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{1}{2} \right] - \left[ -\frac{1}{2}\log(2\pi) - \frac{1}{2}(\sigma^2 + \mu^2) \right]$$
   The $-\frac{1}{2}\log(2\pi)$ terms cancel out:
   $$KL(p \parallel q) = -\frac{1}{2}\log(\sigma^2) - \frac{1}{2} + \frac{1}{2}\sigma^2 + \frac{1}{2}\mu^2$$
   Factoring out $-\frac{1}{2}$:
   $$KL(p \parallel q) = -\frac{1}{2} \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)$$

For $J$ independent latent dimensions, the joint KL divergence is simply the sum:
$$\text{KL Loss} = -\frac{1}{2} \sum_{j=1}^J \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right) \quad \blacksquare$$

---

## 4. Production PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder projects input to latent distribution parameters
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder reconstructs latent sample back to input space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Assume outputs normalized to [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + std * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(reconstructed_x, x, mu, logvar):
    # 1. Term 1: Reconstruction Loss using BCE (assume data in [0,1])
    bce_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    
    # 2. Term 2: Closed-form Gaussian KL Loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return bce_loss + kl_loss
```
