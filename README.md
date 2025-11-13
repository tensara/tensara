
<div align="center">
  <picture>
     <img width="70%" alt="Tensara" src="https://github.com/user-attachments/assets/acd95491-2104-48d4-af84-bf2b23c95e72" />
  </picture>
</div>

[![Twitter](https://img.shields.io/badge/twitter-follow-black?logo=x)](https://x.com/tensarahq) [![GitHub stars](https://img.shields.io/github/stars/tensara/tensara?style=flat&logo=github&color=blue&logoColor=ffffff)](https://github.com/tensara/tensara/stargazers) [![GitHub forks](https://img.shields.io/github/forks/tensara/tensara?style=flat&logo=github&color=green&logoColor=ffffff)](https://github.com/tensara/tensara/network/members)

[Tensara](https://tensara.org/) is a platform for GPU programming challenges in CUDA, Triton, Mojo, etc. Users can write efficient GPU kernels to solve our problems and see how their solutions compare with others on the platform.


https://github.com/user-attachments/assets/96457139-2a27-493c-8352-df5ceb298369




## Features
- **Problems**: Solve 60+ challenges in CUDA, Triton, Mojo, and HIP across multiple difficulty levels.
- **Benchmarking**: Run your solutions on actual GPUs - both NVIDIA (T4, H100, A100) and AMD (MI210, MI250X, MI300X) with precise performance measurement.
- **Leaderboards**: Compare your performance against other developers on per-GPU rankings.
- **Baseline Comparisons**: See how your optimized kernels stack up against PyTorch, Triton, and other framework implementations
- **CLI Tool**: Submit and test solutions directly from your terminal with the Tensara CLI
- **AMD GPU Support**: Write and benchmark ROCm/HIP kernels on real AMD hardware

## GPU Backend Options

Tensara supports both NVIDIA and AMD GPUs through two different backend systems:

### NVIDIA GPUs (Modal)
- **Available GPUs**: T4, A100, H100
- **Deployment**: Serverless via Modal.com
- **Cold Start**: 30-60 seconds
- **Languages**: CUDA, Triton, Mojo
- **Best For**: Quick testing, automatic scaling, CUDA-optimized workloads

### AMD GPUs (dstack + Hot Aisle) 🆕
Tensara now supports AMD GPUs via dstack orchestration! Write and benchmark HIP/ROCm kernels on real AMD hardware:

**Available GPUs:**
- 🚀 **MI210** (64GB) - $3/hr - Cost-effective development
- ⚡ **MI250X** (128GB) - $5/hr - High-performance compute
- 🔥 **MI300A** (192GB) - $7/hr - APU workloads
- ⭐ **MI300X** (192GB) - $8/hr - Latest CDNA3 architecture

**Key Features:**
- 📦 **VM Pooling** - Keep warm VMs for 1-2s response times
- 🔄 **Binary Caching** - Compiled kernels cached for faster iteration
- 📊 **Cost Tracking** - Built-in monitoring and cost reporting
- 🎯 **Compatible API** - Drop-in replacement for Modal backend

**Getting Started:**
- 📖 **[Quick Start Guide (dstack)](QUICKSTART_DSTACK_AMD.md)** - Deploy AMD GPU service with dstack
- 📘 **[Quick Start Guide (Local)](QUICKSTART_AMD_ROCM.md)** - Run locally with ROCm
- 💰 **Low Cost** - Test for $0.01-0.05 per run
- 🛠️ **Full Infrastructure** - Powered by dstack.ai + Hot Aisle

**Perfect For:**
- Learning AMD GPU programming and ROCm/HIP
- Comparing CUDA vs HIP performance
- Optimizing for AMD CDNA architectures
- Building portable GPU code (CUDA → HIP)
- Cost-sensitive workloads (MI210 < NVIDIA A100)

**Comparison:**

| Feature | Modal (NVIDIA) | dstack (AMD) |
|---------|---------------|--------------|
| Cold Start | 30-60s | 5-10 min (first), 1-2s (warm) |
| Languages | CUDA, Triton, Mojo | HIP/ROCm |
| Scaling | Automatic | Manual (VM pooling) |
| GPU Options | T4, A100, H100 | MI210, MI250X, MI300A/X |
| Cost (High-End) | ~$4.5/hr (H100) | ~$8/hr (MI300X) |
| Setup | `modal deploy` | `dstack apply` |

## Contributions

![demo](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcWNqYWt2eGM3ZjR4Mzk0emN6dnFlcW82emM0bTh1c3R2YmZmeWk2ayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YV46Vyr4Tx8j8pHW3R/giphy.gif)

## Sponsors

Thank you to our sponsors who help make Tensara possible:

- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=tensara) - Modal lets you run
jobs in the cloud, by just writing a few lines of Python. Customers use Modal to deploy Gen AI models at large scale,
fine-tune large language models, run protein folding simulations, and much more.

We use Modal to securely run accurate benchmarks on various GPUs.

## Contact

Interested in sponsoring? Contact us at [sponsor@tensara.org](mailto:sponsor@tensara.org) or hit us up [on Twitter](https://x.com/tensarahq)! 
