# Install of XVerseBench

Existing controlled image generation benchmarks often focus on either maintaining identity or object appearance consistency, rarely encompassing datasets that rigorously test both aspects. To comprehensively assess the models' single-subject and multi-subject conditional generation and editing capabilities, we constructed a new benchmark by merging and curating data from DreamBench++ and some generated human images.

Our resulting benchmark XVerseBench comprises 20 distinct human identities, 74 unique objects, and 45 different animal species/individuals. To thoroughly evaluate model effectiveness in subject-driven generation tasks, we developed test sets specifically for single-subject, dual-subject, and triple-subject control scenarios. This benchmark includes 300 unique test prompts covering diverse combinations of humans, objects, and animals. 

<p align="center">
  <img src="../sample/XVerseBench.png" alt="XVerseBench">
</p>
<p align="center"><strong>Figure 1. XVerseBench</strong></p>

The above figure shows more detail information and samples for each categories. For evaluation, we employ a suite of metrics to quantify different aspects of generation quality and control fidelity: including DPG score to assess the model's editing capability, Face ID similarity and DINOv2 similarity to assess the model's preservation of human identity and objects, and Aesthetic Score to measure to evaluate the aesthetics of the generated image. XVerseBench aims to provide a more challenging and holistic evaluation framework for state-of-the-art multi-subject controllable text-to-image generation models.

## Usage

1. Download **DreamBench++** from [https://dreambenchplus.github.io/](https://dreambenchplus.github.io/) and place it into the `data/DreamBench++` directory.
2. Run the following command to rename and segementate the images:
   ```bash
   python assets/rename.py
   python assets/segmentation_sample.py
   ```

## Citation
If XVerseBench is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our paper:
```bibtex
@article{chen2025xverse,
  title={XVerse: Consistent Multi-Subject Control of Identity and Semantic Attributes via DiT Modulation},
  author={Chen, Bowen and Zhao, Mengyi and Sun, Haomiao and Chen, Li and Wang, Xu and Du, Kang and Wu, Xinglong},
  journal={arXiv preprint arXiv:2506.21416},
  year={2025}
}
```


> Disclaimer：
>
> Your access to and use of this dataset are at your own risk. We do not guarantee the accuracy of this dataset. The dataset is provided “as is” and we make no warranty or representation to you with respect to it and we expressly disclaim, and hereby expressly waive, all warranties, express, implied, statutory or otherwise. This includes, without limitation, warranties of quality, performance, merchantability or fitness for a particular purpose, non-infringement, absence of latent or other defects, accuracy, or the presence or absence of errors, whether or not known or discoverable.
> 
> In no event will we be liable to you on any legal theory (including, without limitation, negligence) or otherwise for any direct, special, indirect, incidental, consequential, punitive, exemplary, or other losses, costs, expenses, or damages arising out of this public license or use of the licensed material.
>
> The disclaimer of warranties and limitation of liability provided above shall be interpreted in a manner that, to the extent possible, most closely approximates an absolute disclaimer and waiver of all liability.

