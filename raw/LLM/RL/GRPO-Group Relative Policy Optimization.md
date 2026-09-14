---
title: "GRPO：Group Relative Policy Optimization"
source: "https://zhuanlan.zhihu.com/p/20021693569"
author:
  - "[[知行者大模型全栈算法]]"
published:
created: 2026-09-14
description: "paper: https://arxiv.org/pdf/2402.03300 这里快速介绍一下 deepseek 提出来的这个GRPO 的算法原理。 暂时不对论文通篇进行讲解了。 GRPO的核心思想是通过 组内相对奖励来估计基线（baseline），从而避免使用额外…"
tags:
  - "clippings"
---
目录

收起

1\. 框架图

2\. 算法原理

2.1 PPO 复习

2.2 GRPO的优化

3\. GRPO 计算总结

![](https://pica.zhimg.com/v2-77ab51198a060fbf8f72bd23f449556c_1440w.jpg)

paper: [arxiv.org/pdf/2402.0330](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2402.03300)

这里快速介绍一下 deepseek 提出来的这个 [GRPO](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=1&q=GRPO&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiJHUlBPIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjUzMDI2NzE0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.7XS3Bp4SI0D_vOPh8c2ANSu4ZAWGhHiEOtq6N5dt6WM&zhida_source=entity) 的算法原理。 暂时不对论文通篇进行讲解了。

GRPO的核心思想是通过 **组内相对奖励** 来估计基线（baseline），从而避免使用额外的价值函数模型（critic model）。传统的PPO算法需要训练一个价值函数来估计优势函数（advantage function），而GRPO通过从同一问题的多个输出中计算平均奖励来替代这一过程，显著减少了内存和计算资源的消耗。

## 1\. 框架图

首先看一下PPO 与GRPO 的比较图。 对PPO 算法不熟悉的话，可以查看前一篇文章： [知行者：PPO: Proximal Policy Optimization Algorithms](https://zhuanlan.zhihu.com/p/19949917958)

![](https://pica.zhimg.com/v2-bfd8c156b744b6a9d44819b604ddf19e_1440w.jpg)

从图上可以看出，GRPO 与PPO 的主要区别有：

- GRPO 省略了 value function model.
- GRPO reward 计算，改成了一个q 生成多个r, 然后reward 打分。
- PPO [优势函数](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=2&q=%E4%BC%98%E5%8A%BF%E5%87%BD%E6%95%B0&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiLkvJjlir_lh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTMwMjY3MTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.D-dRzVgsOaezZjPb34ID9ln_MDFdL5ZLzz_Xbvhg05c&zhida_source=entity) 计算时，KL 是包含在 [GAE](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=1&q=GAE&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiJHQUUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTMwMjY3MTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.s8IYuu9NXKamLnxWoIG6YDWXKobX6bkSOGUv4fMOKPM&zhida_source=entity) 内部的。 GRPO 直接挪到了外面，同时修改了计算方法。

## 2\. 算法原理

### 2.1 PPO 复习

直接上原文，首先是PPO 的 [目标函数](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=1&q=%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiLnm67moIflh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTMwMjY3MTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.DFVpHhbXz_JUHl3KR2yW9zbydWzBvLI1HjPsZahn9_w&zhida_source=entity) ：

这个比较熟悉，策略概率比与优势函数的乘积。 同时做了clip限制了参数更新范围。

![](https://pic1.zhimg.com/v2-0e7221bfbed0671dc111989039346f08_1440w.jpg)

公式2 是PPO 中优势函数的计算。 在reward 打分上，加一个per-token 的 [KL 散度](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=1&q=KL+%E6%95%A3%E5%BA%A6&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiJLTCDmlaPluqYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTMwMjY3MTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.rDlca_u2HmJJ0YdaFT6b1C6EolrN3g-DwC1YCJ3qlbA&zhida_source=entity) 惩罚。

![](https://pica.zhimg.com/v2-1935f13bde721f907ad36b63ecfba5f8_1440w.jpg)

### 2.2 GRPO的优化

下面是GRPO 的改进。 **论文认为value function model 占用了额外的显存和计算资源** 。因此提出以下的改进方法。

去除value function, reward 直接对单个q生成的response进行打分， [归一化](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=1&q=%E5%BD%92%E4%B8%80%E5%8C%96&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiLlvZLkuIDljJYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTMwMjY3MTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.das-_dPAyNqw_PKDnfRRJp3nNwRk65ak9cMaPCXQLWQ&zhida_source=entity) 后，作为替代的优势函数。

同时将KL散度抑制，移到了优势函数计算的外面。 KL 散度的计算也进行了改进，可以见公式4. 为了保证 [KL散度](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=2&q=KL%E6%95%A3%E5%BA%A6&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiJLTOaVo-W6piIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI1MzAyNjcxNCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoyLCJ6ZF90b2tlbiI6bnVsbH0.8tgv3gawvA79pKBhM301wamYcDhFn4F-9JGbdv95mNc&zhida_source=entity) 为正值。

![](https://pica.zhimg.com/v2-ffa61975aeb7953a32ca685f48403eb6_1440w.jpg)

![](https://picx.zhimg.com/v2-56f27b571cff5fc485e405ff6da5affb_1440w.jpg)

下图是基于group reward 计算优势函数的，归一化公式：

![](https://pic3.zhimg.com/v2-77583747ab046e8b02dd122f43b6abb0_1440w.jpg)

下面是GRPO 的计算 [伪代码](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=1&q=%E4%BC%AA%E4%BB%A3%E7%A0%81&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiLkvKrku6PnoIEiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTMwMjY3MTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.gDs01q46Z_PtmSL1oHuyISXWDPin-awC1bo1O9pgJ8M&zhida_source=entity) ：

![](https://pica.zhimg.com/v2-a24c5143c977c7c04898e41679eee3c6_1440w.jpg)

GRPO的计算流程包括：

1. 采样一组输出并计算每个输出的奖励。
2. 对组内奖励进行归一化处理。
3. 使用归一化后的奖励计算优势函数。
4. 通过最大化目标函数更新策略模型。
5. 迭代训练，逐步优化策略模型。

GRPO通过组内相对奖励估计基线，避免了传统PPO中 [价值函数](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=3&q=%E4%BB%B7%E5%80%BC%E5%87%BD%E6%95%B0&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiLku7flgLzlh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTMwMjY3MTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MywiemRfdG9rZW4iOm51bGx9.LyEiU0kxImvyTVNvwOd_hYVdvI6AYwsE3JziBFN9kOQ&zhida_source=entity) 的使用，显著减少了训练资源消耗，同时提升了模型在 [数学推理](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=1&q=%E6%95%B0%E5%AD%A6%E6%8E%A8%E7%90%86&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiLmlbDlrabmjqjnkIYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTMwMjY3MTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.Zx64IklzBInMn6a6UCmLNNit4ct_iIyauDlRN6n8zL0&zhida_source=entity) 等复杂任务中的表现。

## 3\. GRPO 计算总结

![](https://pic4.zhimg.com/v2-2f9c1b479bdf22dcd0040f94bc99eb47_1440w.jpg)

![](https://pic3.zhimg.com/v2-090aabaf4e212a2146d110c37d18ceee_1440w.jpg)

- 这里的公式缺了一部分，不要在意。可以查看查看原文截图。
![](https://pica.zhimg.com/v2-3082244e0bd38853ac9b0bb04df7408c_1440w.jpg)

GRPO 的确节约了 [显存](https://zhida.zhihu.com/search?content_id=253026714&content_type=Article&match_order=2&q=%E6%98%BE%E5%AD%98&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk1NTAwODQsInEiOiLmmL7lrZgiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNTMwMjY3MTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.9CWgrIvn3KQmsrHHWlFYjUM2nWDg7Z6ob4n_33a-sf4&zhida_source=entity) 和计算资源。 但是是否真的提升复杂任务能力保留疑问。

编辑于 2025-02-09 16:58・北京[大语言模型](https://www.zhihu.com/topic/27267395)