# LexaLCM-Pre1-400M Two-Tower Latent Diffusion Large Concept Model
これは、Meta FAIRのTwo-Tower Diffusion LCMアーキテクチャを主に基にした、394,900,736個のパラメータを持つ事前学習済みのLCMで、Hugging Face Transformersで実装されています。

[[Meta FAIRのLCMの研究論文（英語）]](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)

It is a pre-trained LCM with 394,900,736 parameters mostly based on Meta FAIR's Two-Tower Diffusion LCM architecture, but in Hugging FaceTransformers.

[[Meta FAIR's LCM Paper]](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)

最初のバージョンは、事前にセグメント化およびコンセプト埋め込みが行われた240万件の日本語および英語のWikipedia記事を用いて学習されています。セグメント化は、1セグメントあたり最大250文字に制限されたSaTを使用して行われ、埋め込みにはSONARが使用されました。
[[データセット]](https://huggingface.co/datasets/Lexa-B/LexaLCM_Datasets)

The first version was trained on a dataset of 2.4M Japanese Wikipedia articles that have been pre-segmented and concept-embedded. Segmentation was performed using SaT capped at 250 characters/segment and embedded was performed with SONAR.
[[Dataset]](https://huggingface.co/datasets/Lexa-B/LexaLCM_Datasets)

## インストール手順 ｜ Installation

```bash
uv venv # create a new virtual environment
source .venv/bin/activate # activate the virtual environment
uv pip install -e ".[gpu]" # install the dependencies (gpu)... if you want to install the dependencies (cpu), use ".[cpu]" instead
```

**Note: You'll need to download the model weights and dataset from the Hugging Face Hub.**
The model weights are located at:
[LexaLCM_Pre1](https://huggingface.co/Lexa-B/LexaLCM_Pre1)

The dataset is located at:
[LexaLCM_Datasets](https://huggingface.co/datasets/Lexa-B/LexaLCM_Datasets) (use the Pre1 version if you specifically want to use the same dataset as the model was trained on)

## Inference
**Note: you'll need to have a GPU with at least 12GB of VRAM to run this as is... I haven't implemented quant at inference features yet.**
```bash
clear & uv run Tests/TestInference.py
```

Currently, it's not very smart, but it's a good start. Expect to see something like this:

```txt
2025-06-01 11:18:23,843 - sonar_pipeline - INFO - Initializing pipeline with config: PipelineConfig(device='cuda', dtype=torch.float32, language='eng_Latn', verbose=True, sequential=False)
2025-06-01 11:18:23,843 - sonar_pipeline - INFO - Initialized TextToEmbeddingPipeline
2025-06-01 11:18:23,843 - sonar_pipeline - INFO - Initializing pipeline with config: PipelineConfig(device='cuda', dtype=torch.float32, language='eng_Latn', verbose=True, sequential=False)
2025-06-01 11:18:23,843 - sonar_pipeline - INFO - Initialized EmbeddingToTextPipeline
2025-06-01 11:18:23,843 - sonar_pipeline - INFO - Encoding sentences: ['[[Start of Text.]]']
2025-06-01 11:18:31,546 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-01 11:18:31,547 - sonar_pipeline - INFO - Encoding sentences: ['The Sengoku era was a period of great conflict in Japan.']
2025-06-01 11:18:36,275 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-01 11:18:36,275 - sonar_pipeline - INFO - Encoding sentences: ['Many clans and their samurai fought in that time.']
2025-06-01 11:18:40,922 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-01 11:18:40,922 - sonar_pipeline - INFO - Encoding sentences: ['It was followed by a period of peace and cultural growth.']
2025-06-01 11:18:45,544 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
→ Context shape: torch.Size([1, 4, 1024]), dtype: torch.float32
[DEBUG - model] labels is None, likely being used for inference. Returning predictied embeddings - shape=torch.Size([1, 4, 1024]), dtype=torch.float32
2025-06-01 11:18:46,206 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-01 11:18:54,853 - sonar_pipeline - INFO - Decoded text: ['He is the founder and chairman of the board of directors of the American Institute of Certified Public Accountants.']
Step 0 model next-token guess: He is the founder and chairman of the board of directors of the American Institute of Certified Public Accountants.
2025-06-01 11:18:54,853 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-01 11:19:00,483 - sonar_pipeline - INFO - Decoded text: ['It\'s called the "City of Dreams".']
Step 1 model next-token guess: It's called the "City of Dreams".
2025-06-01 11:19:00,483 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-01 11:19:06,195 - sonar_pipeline - INFO - Decoded text: ["It's called the Great Wall of China."]
Step 2 model next-token guess: It's called the Great Wall of China.
2025-06-01 11:19:06,195 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-01 11:19:11,882 - sonar_pipeline - INFO - Decoded text: ["I'm not sure what you're saying."]
Step 3 model next-token guess: I'm not sure what you're saying.
[1]+  Done                    clear
```

*Ok, so what's going on here? We'll break it down step by step because there's a lot going on and I'm not going to assume that all readers are fammiliar with 'Attention' and Transformer architectures.*

In this example, the model was provided with a prompt of four sentences and it generated a four sentence response.

It worked by taking the four sentences then passing them trough the SONAR encoder to convert them into four 1024D embeddings.

Those four embeddings are then stacked into a single batch with a sequence length of four, resulting in a tensor of shape [1, 4, 1024].  The sequence is then passed into the model, where each embedding attends to all previous embeddings in the sequence, and then attempts to predict the next embedding in the sequence.

For example, the first embedding (shown in English for simplicity), *[[Start of Text.]]*, passes through the model and attemts to predict the next embedding in the sequence... it has no previous embeddings to attend to, so it just predicts the next embedding in the sequence.

The next embedding, *The Sengoku era was a period of great conflict in Japan.*, passes through the model and attends to the previous embedding, *[[Start of Text.]]*, and then attempts to predict the next embedding in the sequence... it has more information to attend to, so ideally it can predict the next embedding in the sequence better.

This process continues for the remaining two embeddings, *Many clans and their samurai fought in that time.* and *It was followed by a period of peace and cultural growth.*, where each embedding attends to the previous embeddings in the sequence and attempts to predict the next embedding in the sequence.

This means that the model also outputs a tensor of shape [1, 4, 1024], with embedding_0 being the prediction for the first embedding, embedding_1 being the prediction for the second embedding, and so on... as such, for normal applications, the final embedding in the sequence is the 'most useful' as it is the model's attempt to predict the next embedding after the ending of the sentence.

For this inference script, though, all embeddings in the sequence are decoded for better visualization of the model's behavior. So, once this output is generated, all embeddings in the tensor are then passed through the SONAR decoder to convert the embeddings back into text. This is how we see the output that appears as follows:

> [[Start of Text.]] -> He is the founder and chairman of the board of directors of the American Institute of Certified Public Accountants.
>
> The Sengoku era was a period of great conflict in Japan. -> It's called the "City of Dreams".
>
> Many clans and their samurai fought in that time. -> It's called the Great Wall of China.
>
> It was followed by a period of peace and cultural growth. -> I'm not sure what you're saying.

This is a good start, but it's not very smart yet. 





## AIの事前学習手順 ｜ Training

### 事前テストを実行する ｜ Dry run (sanity check) ## ToDo: fix this
```bash
clear & uv run --extra gpu -m src.LexaLCM.Main --dry-run --verbose
```

### 事前学習手順を始める ｜ Run the training
```bash
clear & uv run --extra gpu -m src.LexaLCM.Main -v
```

## Testing

**Currently, this is not working... I'll patch it in Pre2**

### Test the model
```bash
clear & uv run --extra gpu pytest Tests/TestModel.py
```

### Test the data pipeline
```bash
clear & uv run --extra gpu pytest Tests/TestData.py
```
## Special Concepts
These sentences are the equivalent of special tokens in an LLM. They're a quirk of continuous concept embedding space that the model exists within; because there is no discrete separation of tokens, all special signifiers must coinhabit the same 1024D concept embedding spaces as the normal sentences to be translated. there is no separation.
### Start of Text
日本語：

English:
`[[Start of text.]]`

### End of Text
日本語：

English:
`[[End of text.]]`

### Pad
日本語：

English:

### System
日本語：

English:

### Tool
日本語：

English:

### AI
日本語：

English:

### User
日本語：

English:

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



