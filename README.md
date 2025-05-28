# LexaLCM-Pre1-400M Pre-trained Two-Tower Latent Diffusion Large Concept Model
It is a pre-trained LCM with 394,900,736 parameters mostly based on Meta FAIR's Two-Tower Diffusion LCM architecture, but in HF Transformers.

[[Paper]](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)

The first version is being trained on a dataset of 2.4M Japanese and English Wikipedia articles that have been pre-segmented and concept-embedded. Segmentation was performed using SaT capped at 250 characters/segment and embedded was performed with SONAR.
[[Dataset]](https://huggingface.co/datasets/Lexa-B/LexaLCM_Datasets)

## Installation

```bash
uv venv # create a new virtual environment
source .venv/bin/activate # activate the virtual environment
uv pip install -e ".[gpu]" # install the dependencies (gpu)... if you want to install the dependencies (cpu), use ".[cpu]" instead
```





## Training

### Dry run (sanity check) ## ToDo: fix this
```bash
# clear & uv run --extra gpu src/LexaLCM/LCM/Main.py --dry-run --verbose
```

### Run the training
```bash
clear & uv run --extra gpu -m src.LexaLCM.Main
```


## Testing

### Test the model
```bash
clear & uv run --extra gpu pytest Tests/TestModel.py
```

### Test the data pipeline
```bash
clear & uv run --extra gpu pytest Tests/TestData.py
```


## Dataset handling

If you have a dataset in the format of the Meta FAIR "Large Concept Models" paper, you can convert it to the LexaLCM format using the following command:

```bash
clear & uv run --extra data src/Scripts/Data/ConvertMetaParquet.py -i src/_TEMP/DirtyDatasets/ -o src/LexaLCM/Content/Datasets/ -n wikipedia_data_50k
```

where:
- `-i` is the path to the directory with the dataset
- `-o` is the path to the directory to save the converted dataset
- `-n` is the name of the dataset

and in this example, the dataset is called "wikipedia_data_50k" and is located in the directory `src/_TEMP/DirtyDatasets/`. The converted dataset will be saved in the directory `src/LexaLCM/Content/Datasets/` (the default dataset directory for the LexaLCM).












### Verify the embeddings

```bash
uv run --extra data src/Scripts/Data/VerifyEmbeddings.py 
```

where:
- `-d` is the path to the parquet files

For example:
```bash
clear & uv run --extra data src/Scripts/Data/VerifyEmbeddings.py -d src/LexaLCM/Content/Datasets/Wikipedia_Ja
```

### Convert the dataset to the LexaLCM format

```bash
uv run --extra data src/Scripts/Data/ConvertMetaParquet.py
```

where:
- `-d` is the path to the parquet files



### Visualize the dataset

```bash
uv run --extra data src/Scripts/Data/VisualizeDataset.py 
```

Where:
- `-d` is the path to the parquet files
- `-s` is if the dataset is sampled or if all the files are used (sample=True samples 10% of the files)
- `-b` is the batch size for the evaluation process (default is 10)

For example:
```bash
clear & uv run --extra data src/Scripts/Data/VisualizeDataset.py -b 20 -d src/LexaLCM/Content/Datasets/Wikipedia_Ja
```




## Bootstrap the model

```bash
clear & uv run --extra gpu src/LexaLCM/LCM/Utils/BootstrapLCM.py
```



