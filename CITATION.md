# Citation for StreamSplit

## BibTeX

```bibtex
@inproceedings{streamsplit2026,
  title={StreamSplit: Theoretical Guarantees for Edge Audio Learning},
  author={Anonymous},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026},
  url={https://github.com/yourusername/StreamSplit-AAAI}
}
```

## Text

Anonymous. (2026). StreamSplit: Theoretical Guarantees for Edge Audio Learning. 
In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026).

## Abstract

Large-batch contrastive learning, which is the foundation of modern audio 
representation learning, is inherently incompatible with the limitations of 
edge devices. By using a novel distribution-based methodology, StreamSplit 
resolves this incompatibility and makes contrastive learning with small batches 
possible. With theoretical convergence guarantees for edge scenarios, our 
framework presents three new features: (1) streaming contrastive learning that 
operates on embedding distributions instead of individual samples; (2) a hybrid 
loss that combines Laplacian regularization and Sliced-Wasserstein distance to 
maintain representation quality in spite of resource constraints; and (3) 
reinforcement learning-guided computation splitting that dynamically divides 
workload between the edge and server depending on real-time conditions.

## Key Contributions

1. **Streaming Contrastive Framework** - Enables efficient small-batch edge 
   learning by operating on embedding distributions with formal convergence 
   guarantees

2. **Hybrid Loss Function** - Combines Laplacian and Sliced-Wasserstein 
   regularization to preserve representation quality under resource constraints

3. **RL-Based Adaptive Splitting** - Dynamically divides workload between 
   server and edge based on real-time conditions

4. **Empirical Validation** - Achieves near server-level accuracy (within 2.2%) 
   while reducing bandwidth (77.1%), latency (72.6%), and energy (52.3%)

## Contact

For questions about this implementation or the paper, please open an issue on 
the GitHub repository.
