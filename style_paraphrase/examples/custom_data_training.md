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

## How to finetune train the Paraphraser model
