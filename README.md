
# Human assembly intention recognition

## Usage
1. Create conda environment

```
conda create -n HAIR ython=3.11
conda activate HAIR
pip install -r requirements.txt

```

2. Prepare your dataset.

Our model accept the image 112x112. If you want change the resolution, please chanage config file and corresponding parameters. In addition, we have opened our collected dataset in Zenodo, you can access our dataset with the link https://mdx-box.github.io/MCV_Intention/. 

3. We also privide two checkpoints model, you can download them directly and infer the assembly intention by ```python inference.py```. 

    | Fusion position | Checkpoints download| 
    |----------|------------|
    | middle fusion   | [model_4_4](https://drive.google.com/file/d/1t-7Oq3Zk6profQ4hz3e_91yOhIf_lCP_/view?usp=sharing) | 
    | end    fusion   | [model_8_8](https://drive.google.com/file/d/1raO0i7hg5SMTUzjEmDjk5XaW1IlII5MK/view?usp=sharing) | 