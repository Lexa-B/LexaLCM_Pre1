# LexaLCM Pre1 xxxM Pre-trained Large Concept Model
A pre-trained LCM model with xxxM parameters roughly based on Meta FAIR's Two-Tower Diffusion LCM architecture, but in HF Transformers.

[[Paper]](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)

## Installation

```bash
uv venv # create a new virtual environment
source .venv/bin/activate # activate the virtual environment
uv pip install -e ".[gpu]" # install the dependencies (gpu)... if you want to install the dependencies (cpu), use ".[cpu]" instead
```





## Training

### Dry run (sanity check)
```bash
clear & uv run --extra gpu src/LexaLCM/LCM/Main.py --dry-run --verbose
```



## Dataset handling

### Add train/val split column to the dataset

```bash
uv run --extra data src/Scripts/add_train_val_split.py 
```

where:
- `-d` is the path to the parquet files
- `-v` is the validation ratio
- `-s` is the seed

For example:
```bash
clear & uv run --extra data --extra gpu src/Scripts/add_train_val_split.py -d src/LexaLCM/Content/Datasets/Wikipedia_Ja
```

### Visualize the dataset

```bash
uv run --extra data src/Scripts/VisualizeDataset.py 
```

Where:
- `-d` is the path to the parquet files
- `-s` is if the dataset is sampled or if all the files are used (sample=True samples 10% of the files)
- `-b` is the batch size for the evaluation process (default is 10)

For example:
```bash
clear & uv run --extra data src/Scripts/VisualizeDataset.py -b 20 -d src/LexaLCM/Content/Datasets/Wikipedia_Ja
```