---
title: "TimesFM XReg Modes"
aliases: ["TimesFM XReg", "Residual Covariate Modeling"]
date: 2026-07-23
tags: ["entity", "time-series", "decoder-only", "covariates", "residual"]
sources: ["[[wiki/research/In-Context Learning and Future Covariates in Decoder-Only Forecasting.md]]"]
---

# TimesFM XReg Modes

**TimesFM XReg Modes** are dual decoupled integration strategies implemented in Google's **TimesFM 2.5** (Time Series Foundation Model) `xreg_lib.py` library. Since TimesFM is an autoregressive single-variable (univariate) Decoder model, the `XReg` (Exogenous Regressor) framework provides an elegant way to incorporate multi-dimensional future known covariates without modifying the internal transformer layers of the foundation model.

---

## 1. `xreg + timesfm` Mode (Residual forecasting)

This is the **default and highest-performing mode** in most real-world scenarios. It treats the covariates as primary drivers and leverages the foundation model to forecast pure non-linear residuals.

### **Mechanism Steps:**
1. **Linear Exogenous Fitting**: A light Ridge regression model fits the historical covariates $X_{1:T}$ to target series $y_{1:T}$ and projects the linear trend onto the future horizon:
   $$\hat{y}^{xreg}_{T+1:T+H} = \mathbf{W}^{\top} X_{T+1:T+H} + \mathbf{b}$$
2. **De-biasing**: The historical target values are stripped of covariate-explainable trends, calculating the clean temporal residual sequence:
   $$y^{resid}_{1:T} = y_{1:T} - \hat{y}^{xreg}_{1:T}$$
3. **TimesFM Inference**: The clean residual $y^{resid}_{1:T}$ is fed into TimesFM, which predicts the pure non-linear future trend:
   $$\hat{y}^{timesfm}_{T+1:T+H} = \operatorname{TimesFM}(y^{resid}_{1:T})$$
4. **Summation**: The final forecast is the sum of both components:
   $$\hat{y}_{T+1:T+H} = \hat{y}^{timesfm}_{T+1:T+H} + \hat{y}^{xreg}_{T+1:T+H}$$

*   **Best For**: Datasets with strong, sudden external shocks (e.g., promotional campaigns, black friday extreme sales, holiday power drops). By stripping out these shocks first, the input fed to the foundation model is highly smoothed, preventing model divergence.

---

## 2. `timesfm + xreg` Mode (Error correction)

This mode treats the foundation model as the master predictor and uses covariates as a late-stage error-corrector.

### **Mechanism Steps:**
1. **TimesFM Base Forecast**: Directly feeds raw history $y_{1:T}$ to TimesFM, generating a base future forecast:
   $$\hat{y}^{timesfm}_{T+1:T+H} = \operatorname{TimesFM}(y_{1:T})$$
2. **Error Tracking**: Tracks the base model's step-wise historical prediction errors:
   $$e_{1:T} = y_{1:T} - \hat{y}^{timesfm\_train}_{1:T}$$
3. **Error Modeling**: Fits a linear regression mapping covariates to the historical error:
   $$e_{1:T} \approx \mathbf{W}_{err}^{\top} X_{1:T} + \mathbf{b}_{err}$$
4. **Correction**: Uses future covariates to predict the future error term and adds it to the TimesFM base forecast:
   $$\hat{y}_{T+1:T+H} = \hat{y}^{timesfm}_{T+1:T+H} + \hat{e}^{xreg}_{T+1:T+H}$$

*   **Best For**: Highly stable, regular time series with weak covariate signals (e.g., slight temperature fluctuations affecting seasonal residential power load).

---

## See Also
- [[Chronos-2 Summary]]
- [[CITRAS Summary]]
- [[KV Shift]]
- [[Attention Score Smoothing]]
