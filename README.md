<p align="center">

  <h2 align="center">InstructMoLE: Instruction-Guided Mixture of Low-rank Experts<br>for Multi-Conditional Image Generation</h2>
  <p align="center">
      <a href="https://scholar.google.com/citations?user=ITSm2LYAAAAJ&hl=en">Jinqi Xiao</a><sup>1,2</sup>
      ·
      <a href="https://scholar.google.com/citations?hl=en&user=0TIYjPAAAAAJ&view_op=list_works&sortby=pubdate">Qing Yan</a><sup>1</sup>
    ·  
      <a href="https://liming-jiang.com/">Liming Jiang</a><sup>1</sup>
    ·  
      <a href="https://scholar.google.com/citations?user=-H18WY8AAAAJ&hl=en">Zichuan Liu</a><sup>1</sup>
    ·  
      <a href="https://scholar.google.com/citations?user=VeTCSyEAAAAJ&hl=en">Hao Kang</a><sup>1</sup>
    ·  
      <a href="https://scholar.google.com/citations?user=DkkblQQAAAAJ&hl=en">Shen Sang</a><sup>1</sup>
    ·  
      <a href="https://tiancheng-zhi.github.io/">Tiancheng Zhi</a><sup>1</sup>
    ·  
       <a href="https://www.jingliu.net/">Jing Liu</a><sup>1</sup>
    ·  
      <a href="https://www.yangcheng.site/">Cheng Yang</a><sup>1</sup>
    ·  
      <a href="https://scholar.google.com/citations?user=mFC0wp8AAAAJ&hl=en">Xin Lu</a><sup>1</sup>
    ·  
     <a href="https://sites.google.com/site/boyuaneecs">Bo Yuan</a><sup>2</sup>
    <br>
    <br>
    <sup>1</sup>ByteDance Inc. &nbsp;<sup>2</sup>Rutgers University &nbsp;
    <br>
    </br>
        <a href="https://arxiv.org/abs/2512.21788">
        <img src='https://img.shields.io/badge/arXiv-2512.21788-red' alt='Paper PDF'></a>
     </br>
</p>


**InstructMoLE** (Instruction-Guided Mixture of Low-rank Experts) addresses task interference in multi-conditional image generation by aligning expert selection with global user intent. Unlike standard per-token routing mechanisms that can cause semantic and spatial artifacts, InstructMoLE introduces a unified routing strategy that ensures consistent expert choices across the entire image, enabling effective handling of diverse conditional generation tasks including single image editing, multi-subject generation, and spatial alignment.

<br>

## Key Contributions

**InstructMoLE** solves task interference in multi-conditional image generation through two key innovations:

*   **Instruction-Guided Routing (IGR):** Replaces standard per-token routing with a single, global signal derived from the user's instruction. This enforces a consistent expert choice across the entire image, preventing semantic and spatial artifacts that arise from inconsistent routing decisions.

*   **Output-Space Orthogonality Loss:** A novel regularizer that forces experts to be functionally distinct. It directly prevents expert collapse by penalizing redundant outputs, ensuring effective specialization across different conditional generation tasks.

<br>

## Installation

```bash
conda create -n instruct_mole python=3.11
conda activate instruct_mole
bash install_env.sh
```

The installation script will install all required dependencies including PyTorch, Diffusers, Transformers, and other necessary packages for training and evaluation.

<br>

## Dataset Preparation

For training InstructMoLE, we support multiple open-source datasets covering different conditional generation scenarios:

- **Single Image Editing:** [OmniEdit](https://github.com/stepfun-ai/Step1X-Edit)
- **Multi-Subjects:** [MUSAR-Gen](https://github.com/musar-gen/musar-gen)
- **Subject and Spatial Alignment:** [SubjectSpatial200K](https://huggingface.co/datasets/Xuan-World/SubjectSpatial200K)
- **Spatial Alignment:** [COCO 2017](https://cocodataset.org/)

Please prepare your datasets according to the expected format and place them in the appropriate directories. You can also use your own additional data for model training.

<br>

## Training

To train InstructMoLE, use the provided training script:

```bash
bash run.sh
```

The training script uses `accelerate launch` for distributed training. You can customize training parameters by modifying `train_config.json`, which includes:

### Configuration Parameters

**MoE Configuration:**
- `num_experts`: Number of experts in the mixture (default: 8)
- `num_experts_per_tok`: Number of experts activated per token (default: 4)
- `rank`: Low-rank decomposition rank for experts (default: 32)
- `alpha`: Scaling factor for expert outputs (default: 32)
- `type_aux_loss_alpha`: Weight for type-based auxiliary loss (default: 0.1)
- `token_aux_loss_alpha`: Weight for token-based auxiliary loss (default: 0.01)
- `orthogonal_reg_alpha`: Weight for orthogonality regularization (default: 0.01)
- `use_type_embedding`: Whether to use instruction-guided routing (default: true)

**LoRA Configuration:**
- `r`: LoRA rank (default: 256)
- `lora_alpha`: LoRA alpha scaling factor (default: 256)
- `target_modules`: List of modules to apply LoRA


For more details on training, refer to `train_kontext.py` and `train_config.json`.

<br>

## Evaluation

InstructMoLE supports evaluation on multiple benchmarks:

- **XVerseBench:** Multi-subject conditional generation benchmark
- **OmniContext:** Image editing benchmark
- **Spatial Alignment:** Pose, depth, and canny edge evaluation

Evaluation scripts are provided in the `eval/` directory. Please refer to the respective evaluation scripts for detailed usage instructions.

<br>

## BibTeX

If you find [InstructMoLE](https://arxiv.org/abs/2512.21788) useful for your research and applications, please cite InstructMoLE using this BibTeX:

```bibtex
@misc{xiao2025instructmoleinstructionguidedmixturelowrank,
      title={InstructMoLE: Instruction-Guided Mixture of Low-rank Experts for Multi-Conditional Image Generation}, 
      author={Jinqi Xiao and Qing Yan and Liming Jiang and Zichuan Liu and Hao Kang and Shen Sang and Tiancheng Zhi and Jing Liu and Cheng Yang and Xin Lu and Bo Yuan},
      year={2025},
      eprint={2512.21788},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.21788}, 
}
```

<br>
