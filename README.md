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
- Model: https://huggingface.co/HWatervalley/TiToHe_tnypixel

# Zero-shot
- Download Ollama
- Download local LLaVA and LLaMA3 models
- Create custom LLaMA3 model file - See "contro" for Parameters
- 
  
