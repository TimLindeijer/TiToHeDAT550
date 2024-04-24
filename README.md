# Check-Worthiness using Multi-Modal content

## Folder Structure
```
├───baselines
├───clip_only_90_epochs_kfold
├───clip_only_90_epochs_kfold_woconf
├───clip_only_90_epochs_kfold_wo_conf
├───data
│   └───CT23_1A_checkworthy_multimodal_english_v2
│       └───images_labeled
│           ├───dev
│           ├───dev_test
│           └───train
├───format_checker
├───outputs
├───results_roberta
├───scorer
├───submission-samples
└───tmp_trainer
```

# Fine-tuned
- Google colab with T4 GPU
- Import Autotrain - advanced with pre requisites
- PARAMETERS: !autotrain llm --train --project-name XXX --model TinyPixel/Llama-2-7B-bf16-sharded --data-path . --use-peft --quantization int4 --lr 2e-4 --batch-size 12 --epochs 3 --trainer sft --target-modules q_proj,v_proj --push-to-hub --token XXX --repo-id XXX
- -Inference
- Adapter: https://huggingface.co/HWatervalley/TiToHe_mistral_model
- NOTE: mistral was the first model tested with and the name in the link is a remnant from that
- Model: https://huggingface.co/HWatervalley/TiToHe_tnypixel

# Zero-shot
- Download Ollama
- Download local LLaVA and LLaMA3 models
- Create custom LLaMA3 model - See "contro" for Parameters
- Pre process text, get id,text and label
- Pre process image, get id and encode images to base64
- Pass encodings through LLaVA model to get description
- Match image data with text data through id
- Pass description and text through contro to get model label
- Handle exceptoions i.e model label =  "I will not provide information that could be used for identity theft" manualy labeled as "yes"
- Bootstrap metrics were done with 548 samples, 100 itterations

# Few-shot
Same process as for Zero-shot, but using the few-shot modelfile
  
## Steps to run CLIP or RoBERTa on slurm

This guide will guide you through the steps to run CLIP or RoBERTa on slurm.

### Step 1: Connect to the slurm server
```bash
ssh <username>@ssh1.ux.uis.no
ssh gorina11
```

### Step 2: Clone the repository
```bash
git clone https://github.com/TimLindeijer/TiToHeDAT550.git
cd TiToHeDAT550
```

### Step 3: Get the data

The data is available at [this link](https://gitlab.com/checkthat_lab/clef2023-checkthat-lab/-/tree/main/task1/data). Download the data and place it in the `data` folder.

### Step 4: Setup the conda environment
```bash
sbatch setup.sh
```

### Step 5: Run the script

Before running `start.sh`, make sure to change the job-name and output file name in the script. You should also comment out any of the python scripts that you do not want to run, and uncomment the one you want to run. After that, run the following command:
```bash
sbatch start.sh
```

### Step 6: Check if the job is running
```bash
squeue -p gpuA100
```

### Step 7: Check the output
When the job is running run the following command to check the output:
```bash
tail -f <output_file>
```

## Check out already run scripts
The scripts that have already been run can be found in the `outputs` folder. The scripts are named according to what they have trained/tested with.