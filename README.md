<div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin: 20px 0;">
  <img src="figs/logo.png" alt="DREAM Logo" width="40" />
  <h1 style="margin: 0; font-size: 2em;">DREAM</h1>
</div>

## Contents

- [Setup & Installation](#setup--installation)
- [Inference](#inference)
  - [With UI](#with-ui)
  - [With Code](#with-code)
- [Train](#train)
  - [Generate Train Data](#generate-train-data)
  - [Train the Auto-regression Head](#train-the-auto-regression-head)
  - [Inference on custom models](#inference-on-custom-models)
- [Evaluation](#evaluation)


## Setup & Installation


```bash
cd DREAM
pip install -r requirements.txt
```

## Inference
The inference code we provide automatically allocates model weights (loading a model across multiple GPUs), allowing you to run models that exceed the memory of a single GPU.

### With UI
We have provided a suggested web interface, which you can use by running the following command. After the model is fully loaded, a URL will be output in the terminal, which you can enter into your browser to access.
```bash
python -m dream.application.webui --ea-model-path /home/apc/models/DREAM-Vicuna-7B-v1.3 --base-model-path /home/apc/models/vicuna-7b-v1.3 --model-type vicuna --total-token 8
```
The *total-token* is the number of draft tokens. For smaller models and advanced GPUs, this value can be set larger. Adjusting according to the specific device and model can achieve better results. If set to -1, DREAM will automatically configure this parameter.

### With Code
You can use our provided "eagenerate" for speedup generation just like using 'generate' from Hugging Face. Here is an example.
```python
from dream.model.ea_model import EaModel
from fastchat.model import get_conversation_template
model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=DREAM_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()
your_message="Hello"
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
output=model.tokenizer.decode(output_ids[0])
```


## Train

### Generate Train Data
You can run the following command to generate the training data.
```bash
python -m dream.ge_data.allocation --outdir [path of data]
```
### Train the Auto-regression Head

```bash
cd dream/model
deepspeed main_deepspeed.py --deepspeed_config /home/apc/DREAM/dream/train/ds_config.json --tmpdir /home/apc/Bingle/data/llava_vicuna_mmt_0/12_data/sharegpt_0_7999_mufp16 --cpdir /home/apc/DREAM/dream/train/vicuna-7b-ckpt --configpath /home/apc/DREAM/dream/train/vicuna_7B_config.json
```

### Inference on custom models

If the original LLM structure differs from LLaMA and Mixtral, you can utilize DREAM as follows:

Copy the modeling_basemodelname.py from the Transformers library and proceed to make modifications to leverage the pre-allocated kv_cache for enhanced speed in the base model. You can refer to model/modeling_llama_kv.py for guidance, where places that require modifications are annotated with # [MODIFIED]. These modifications are minimal.


## Evaluation
You can test the speed of DREAM on MT-bench using the following command.
```bash
python -m dream.evaluation.gen_ea_answer_vicuna(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of DREAM weight]\ 
		 --base-model-path [path of the original model]\
```
If you need specific acceleration ratios, you will also need to run the following command to get the speed of vanilla auto-regression.
```bash
python -m dream.evaluation.gen_baseline_answer_vicuna\
		(or gen_ea_answer_vicuna_llama2chat)\
		 --ea-model-path [path of DREAM weight]\ 
		 --base-model-path [path of the original model]\
```
The above two commands will each generate a .jsonl file that records the generation results and wall time. Then, you can use evaluation/speed.py to calculate the ratio of speeds.


## Acknowledgements

This project has been influenced by many excellent projects in the LLM community, such as [Medusa](https://github.com/FasterDecoding/Medusa), [EAGLE](https://github.com/SafeAILab/EAGLE), [FastChat](https://github.com/lm-sys/FastChat), and others. The logo is designed by GPT-4. We also appreciate many valuable discussions with Tianle Cai, Hao Zhang, Ziteng Sun, and others.
