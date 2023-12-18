# How to Adapt STRAP to a Custom Dataset

## Train an Inverse Paraphraser model for your new style dataset

Create a folder in `datasets` which will contain `new_dataset` as `datasets/new_dataset`. Paste your plaintext train/dev/test splits into this folder as `train.txt`, `dev.txt`, `test.txt`. Use one instance per line (note that the model truncates sequences longer than 50 subwords). Add `train.label`, `dev.label`, `test.label` files (with same number of lines as `train.txt`, `dev.txt`, `test.txt`). These files will contain the style label of the corresponding instance. See [this folder](https://drive.google.com/drive/folders/1a7SS3n9Ds3PEcDH7o3rZnWr-AAkVKYZw?usp=sharing) for examples of label files.

1. To convert a plaintext dataset into it's BPE form run the command,

```
python datasets/dataset2bpe.py --dataset datasets/new_dataset
```

Note that this process is reversible. To convert a BPE file back into its raw text form: `python datasets/bpe2text.py --input <input> --output <output>`.

2. Next, for converting the BPE codes to `fairseq` binaries and building a label dictionary, first make sure you have downloaded RoBERTa and setup the `$ROBERTA_LARGE` global variable in your `.bashrc` (see "Setup" for more details). Then run,

```
datasets/bpe2binary.sh datasets/new_dataset
```

3. To train inverse paraphrasers you will need to paraphrase the dataset. First, download the pretrained model `paraphraser_gpt2_large` from [here](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing). After downloading the pretrained paraphrase model run the command,

```
python datasets/paraphrase_splits.py --dataset datasets/new_dataset
```

4. Add an entry to the `DATASET_CONFIG` dictionary in [`style_paraphrase/dataset_config.py`](style_paraphrase/dataset_config.py), customizing configuration if needed.

```
"datasets/new_dataset": BASE_CONFIG
```

5. Enter your dataset in the hyperparameters [file](https://github.com/martiansideofthemoon/style-transfer-paraphrase/blob/master/style_paraphrase/hyperparameters_config.py#L23) and run `python style_paraphrase/schedule.py`.

6. Run the following `.sh` file adapted to the name of your custom dataset to train an inverse paraphraser model

```
#!/bin/sh
#SBATCH --job-name=finetune_gpt2_custom_0
#SBATCH -o style_paraphrase/logs/log_custom_0.txt
#SBATCH --time=167:00:00
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=50GB
#SBATCH -d singleton

#export DATA_DIR=datasets/test_custom
export DATA_DIR=datasets/isaiah

BASE_DIR=style_paraphrase

# Trains the InverseParaphraser and saves to 'model_custom_isaiah'
python3 -m torch.distributed.launch --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/model_custom_isaiah \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-large \
    --do_train \
    --data_dir=$DATA_DIR \
    --save_steps 500 \
    --logging_steps 20 \
    --save_total_limit -1 \
    --evaluate_during_training \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --per_gpu_train_batch_size 5 \
    --job_id custom_isaiah \
    --learning_rate 5e-5 \
    --prefix_input_type paraphrase_250 \
    --global_dense_feature_list none \
    --specific_style_train 0 \
    --optimizer adam \
    --overwrite_output_dir
```

## How to use your custom inverse paraphraser model
1. Get your text that you want to style transfer to your new custom datset style
2. Run the following python code

```
python3 style_paraphrase/evaluation/scripts/style_transfer.py --input_file=datasets/<input_text>.txt --output_file=datasets/<output_filename>.txt --paraphrase_model=paraphraser_gpt2_large --style_transfer_model=style_paraphrase/saved_models/<custom_dataset_model>/<checkpoint_name> --detokenize --post_detokenize --top_p=0.0
```
3. It is saved in `</path/to/style_transfer_model>/eval_nucleus_paraphrase_0.0/`

## How to finetune train the Paraphraser model
1. Build a `.tsv` file of paired sentences with the following code
```
def create_tsv_from_txt(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            sentence = line.strip()  # Remove any leading/trailing whitespace
            outfile.write(f"{sentence}\t{sentence}\n")

# Use the function with your file paths
create_tsv_from_txt('bom_cleaned.txt', 'bom.tsv')
```

2. Extract a `train.pickle` and `dev.pickle` training data from the `.tsv` file

```
python3 datasets/prepare_paraphrase_data.py --input_file=datasets/bom/bom.tsv --output_folder=datasets/custom_bom_finetuned_paraphraser --train_fraction=0.95
```

3. Fine-tune train the paraphraser

```
#!/bin/sh
#SBATCH --job-name=finetune_gpt2_test_paraphrase
#SBATCH -o style_paraphrase/logs/log_test_paraphrase.txt
#SBATCH --time=167:00:00
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=50GB
#SBATCH -d singleton

# Experiment Details :- GPT2-large model for paraphrasing.
# Run Details :- accumulation = 2, batch_size = 5, beam_size = 1, cpus = 3, dataset = datasets/paranmt_filtered, eval_batch_size = 1, global_dense_feature_list = none, gpu = m40, learning_rate = 5e-5, memory = 50, model_name = gpt2-large, ngpus = 1, num_epochs = 3, optimizer = adam, prefix_input_type = original, save_steps = 500, save_total_limit = -1, specific_style_train = -1, stop_token = eos

export DATA_DIR=datasets/custom_bom_finetuned_paraphraser #paranmt_filtered

# source style-venv/bin/activate

BASE_DIR=style_paraphrase

python3 -m torch.distributed.launch --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/bom_finetuned_paraphraser \
    --model_type=gpt2 \
    --model_name_or_path=paraphraser_gpt2_large \
    --do_train \
    --data_dir=$DATA_DIR \
    --save_steps 500 \
    --logging_steps 20 \
    --save_total_limit -1 \
    --evaluate_during_training \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --per_gpu_train_batch_size 5 \
    --job_id bom_finetuning_paraphraser \
    --learning_rate 5e-5 \
    --prefix_input_type original \
    --global_dense_feature_list none \
    --specific_style_train -1 \
    --optimizer adam
```

## Contextual Embeddings (novel approach)

1. Clean Isaiah context from church website (https://www.churchofjesuschrist.org/study/manual/old-testament-student-manual-kings-malachi/enrichment-e?lang=eng)

2. Fine-tune train Paraphraser model with this new Isaiah context (see steps above)
3. Build an Inverse Paraphraser model by fine-tune training the English_1990 model with this new Isaiah context and improved Paraphraser model into modern day English
