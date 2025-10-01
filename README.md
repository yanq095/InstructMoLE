<div align="center">
<h1> InstructMoLE </h1>

<h3>InstructMoLE: Instruction-Guided Mixture of Low-rank Experts for Multi-Conditional Image Generation</h3>
Jinqi Xiao, Qing Yan, Liming Jiang, Zichuan Liu, Hao Kang, Shen Sang, Tiancheng Zhi, Jing Liu, Cheng Yang, Xin Lu, Bo Yuan
<br>
<a href=""><img src="" alt="arXiv"></a>
</div>

## 🌠 Key Features

**InstructMOLE** solves task interference in multi-conditional image generation by aligning expert selection with global user intent.

*   **Instruction-Guided Routing (IGR):** Replaces standard per-token routing with a single, global signal derived from the user's instruction. This enforces a consistent expert choice across the entire image, preventing semantic and spatial artifacts.

*   **Output-Space Orthogonality Loss:** A novel regularizer that forces experts to be functionally distinct. It directly prevents expert collapse by penalizing redundant outputs, ensuring effective specialization.
## 🔧 Dependencies and Installation

```bash
conda create -n instruct_mole python=3.11
conda activate instruct_mole
bash install_env.sh
```

## 🧩 Prepare Dataset
For your reference, we've included several open-source datasets below. Feel free to use your own additional data for model training.
 - Single Image Editing: OmniEdit
 - Multi-Subjects: MUSAR-Gen
 - Subject and Spatial Alignment: SubjectSpatial200K
 - Spatial Alignment: coco2017


## 🧩 Training
```bash
bash run.sh
```

## 📚 Citation
```

```
