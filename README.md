# Understanding Long Videos in One Multimodal Language Model Pass

[Kanchana Ranasinghe](https://scholar.google.com/citations?user=K2WBZTwAAAAJ&hl=en),
[Xiang Li](https://scholar.google.com/citations?user=qkyC7KQAAAAJ&hl=en),
[Kumara Kahatapitiya](https://scholar.google.com/citations?user=ExGkzjQAAAAJ&hl=en), 
[Michael Ryoo](https://scholar.google.com/citations?user=vcw0TJIAAAAJ&hl=en)

**[Paper Link]()** | **[Project Page](https://kahnchana.github.io/mvu)** 


## Overview
> **Abstract:**
>*Large Language Models (LLMs), known to contain a strong awareness of world knowledge, have allowed recent approaches to achieve excellent performance on Long-Video Understanding benchmarks, but at high inference costs. 
In this work, we first propose Likelihood Selection, a simple technique that unlocks faster inference in autoregressive LLMs for multiple-choice tasks common in long-video benchmarks.
In addition to faster inference, we discover the resulting models to yield surprisingly good accuracy on long-video tasks, even with no video specific information. 
Building on this, we inject video-specific object-centric information extracted from off-the-shelf pre-trained models and utilize natural language as a medium for information fusion. Our resulting Multimodal Video Understanding (MVU) framework demonstrates state-of-the-art performance across long-video and fine-grained action recognition benchmarks.*

<p align="center">
  <img alt="intro_image" src=".github/intro.png" width="950"/>
</p>
<p align="center">
We propose three variants of our framework. (left-top) Just LLM only world knowledge with zero task-specific awareness. (left-right) Single Frame VLM processes an additional center frame to obtain task context but accesses no video specific information. (right) Our complete approach, MVU extracts three additional object-centric modalities followed by fusion in language space. LS refers to likelihood selection. 
</p>


## Quickstart 

We provide two notebooks to explore our two modality-constrained variants. These models require only an LLM / VLM to operate and can be setup easily. Only the python dependencies listed at top of each notebook need to be installed in a `Python=3.8` environment. Use two different environments for two notebooks (some LLMs require latest HF version incompatible with LLaVA used for VLM). All data will be downloaded automatically (less than 5MB for LLM only / approximately 100MB for SF-VLM). 

* LLM Only: [notebook](notebooks/llm_only.ipynb) 
* Single Frame VLM: [notebook](notebooks/sf_vlm.ipynb) 

The following results on EgoSchema (500 video subset) dataset can be replicated using our notebooks.
| Method   |       Backbone      | Acc (%) | Time (s) |
|----------|:-------------------:|:-------:|:--------:|
| LLM Only |   Llama-2-7b-Chat   |   17.4  |   0.72   |
| LLM Only |     Gemma-7b-IT     |   45.8  |   1.84   |
| LLM Only | Mistral-7B-Instruct |   45.8  |   0.41   |
| SF-VLM   |    LLaVA-v1.5-13B   |   55.8  |   1.70   |

Our full MVU framework requires EgoSchema videos for inference and involves multiple pretrained models. Refer to next section for using it. 

## Likelihood Selection 
Our proposed Likelihood Selection (LS) strategy for long-video understanding tasks is a standalone function that can be incorporated with other LLM-based frameworks. Two working examples of LS are presented in each of our notebooks in the above 
section.

Given access to network logits, LS can easily be implemented. We refer the reader to our [`calc_loglikelihood`](https://github.com/kahnchana/mvu/blob/master/notebooks/utils.py#L37) method in 
`notebooks/utils.py` for the PyTorch implementation. Note that when applying in a different task, this selection setup may be sensitive to the prompt nature and could require some handcrafting of the the textual prompts used to query the model (as is common for most LLM based setups). 


## Installation 

1. Clone our repository
   ```
   git clone https://github.com/kahnchana/mvu.git
   ```
2. Create conda environment
    ```
    conda create -n mvu python=3.8
    conda activate mvu
    ```
3. Install python dependencies
    ```
    pip install -r requirements.txt
    ```

## Dataset Preparation
Our main evaluations utilize three datasets: EgoSchema, NextQA, and Open X-Embodiment. We direct to their websites for dataset setup. 

1. [EgoSchema](https://github.com/egoschema/EgoSchema)
2. [NextQA](https://github.com/doc-doc/NExT-QA)
3. [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment)

We follow the default instructions in their websites to download these datasets. We describe the dataset splits used for each evaluation in our paper. 


## MVU Framework

We now detail our MVU framework. This is built over the Single Frame VLM variant. 

### Frame Selection 

```
python src/model_frame_selection.py
```

### Object Centric Modalities
We provide the pre-extracted data for each modality along with the templates used for language based fusion.
These will be automatically download in the following scripts. 

### Long Video QnA
Modify the name of the dataset (EgoSchema, NextQA) and the data root (directory where the dataset was downloaded).
```
python src/model_video_infer.py --dataset $DATASET --data-root $DATA_ROOT
```


## References
Our work builds over the [LLaVA codebase](https://github.com/haotian-liu/LLaVA/tree/main) and utilizes multiple pretrained models from [HuggingFace](https://huggingface.co) (HF).  From HF, we use three different LLMs: [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [LLAMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), and [Gemma-7B](https://huggingface.co/google/gemma-7b-it). We also use [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) for object detection. We thank all authors and maintainers of above codebases for their valuable open-source contributions.

## Citation
If you find our work or code useful, please consider citing our paper and leaving a star on our repo. 
```
@misc{rana2024mvu,
      title={Understanding Long Videos in One Multimodal Language Model Pass}, 
      author={Kanchana Ranasinghe and Xiang Li and Kumara Kahatapitiya and Michael Ryoo},
      year={2024},
}
```
