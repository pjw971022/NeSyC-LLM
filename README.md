# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks in Open Domains

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openreview.net/pdf?id=VoayJihXra)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-green)](https://openreview.net/forum?id=VoayJihXra)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/pjw971022/nesyc-LLM)

## Authors
- [Jinwoo Park](https://pjw971022.github.io/)<sup>1*</sup>
- [Wonje Choi](https://scholar.google.com/citations?user=L4d1CjEAAAAJ&hl=ko)<sup>1*</sup>
- [Sanghyun Ahn](https://scholar.google.co.kr/citations?user=xGh7hdIAAAAJ&hl=ko)<sup>1</sup>
- [Daehee Lee](https://www.linkedin.com/in/daehee-lee-10b396246/?locale=en_US)<sup>1,2</sup>
- [Honguk Woo](https://scholar.google.co.kr/citations?user=Gaxjc7UAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup>Sungkyunkwan University (SKKU), <sup>2</sup>Carnegie Mellon University (CMU)  
<sup>*</sup>Equal contribution

## Abstract

We explore neuro-symbolic approaches to generalize actionable knowledge, enabling embodied agents to tackle complex tasks more effectively in open-domain environments. A key challenge for embodied agents is the generalization of knowledge across diverse environments and situations, as limited experiences often confine them to their prior knowledge.

To address this issue, we introduce a novel framework, **NeSyC**, a neuro-symbolic continual learner that emulates the hypothetico-deductive model by continuously formulating and validating knowledge from limited experiences through the combined use of Large Language Models (LLMs) and symbolic tools. Specifically, NeSyC incorporates a **contrastive generality improvement scheme**. This scheme iteratively produces hypotheses using LLMs and conducts contrastive validation with symbolic tools, reinforcing the justification for admissible actions while minimizing the inference of inadmissible ones.

We also introduce a **memory-based monitoring scheme** that efficiently detects action errors and triggers the knowledge evolution process across domains. Experiments conducted on embodied control benchmarks—including ALFWorld, VirtualHome, Minecraft, RLBench, and a real-world robotic scenario—demonstrate that NeSyC is highly effective in solving complex embodied tasks across a range of open-domain settings.

## Key Features

- Neuro-symbolic continual learning framework
- Contrastive generality improvement scheme
- Memory-based monitoring for error detection
- Effective knowledge evolution across domains
  
## Citation

```bibtex
@inproceedings{choinesyc,
  title={NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks in Open Domains},
  author={Choi, Wonje and Park, Jinwoo and Ahn, Sanghyun and Lee, Daehee and Woo, Honguk},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
