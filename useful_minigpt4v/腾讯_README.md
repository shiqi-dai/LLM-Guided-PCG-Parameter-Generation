# 配置

详情见 README.md （miniGPT-v原始工程中的说明文档）

- 1、下载 LLM 的 weights 到到自定义目录 /path/LLM；
- 2、修改对应的 evaluation .yaml 文件中填入 LLM weights  的自定义目录

- 3、下载 miniGPT-4 （或者 miniGPT-v2）的 weights 到自定义目录 /path/minigpt4v；
- 4、修改对应的 minigptv2_eval.yaml文件中填入 LLM weights  的自定义目录

### LLM weights 下载 & 修改.yanl

**MiniGPT-v2** is based on Llama2 Chat 7B. For **MiniGPT-4**, we have both Vicuna V0 and Llama 2 version.
Download the corresponding LLM weights from the following huggingface space via clone the repository using git-lfs.

|                            Llama 2 Chat 7B                             |                                           Vicuna V0 13B                                           |                                          Vicuna V0 7B                                          |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
[Download](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) | [Downlad](https://huggingface.co/Vision-CAIR/vicuna/tree/main) | [Download](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) 


Then, set the variable *llama_model* in the model config file to the LLM weight path.

* For MiniGPT-v2, set the LLM path 
[here](minigpt4/configs/models/minigpt_v2.yaml#L15) at Line 14.

* For MiniGPT-4 (Llama2), set the LLM path 
[here](minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15.

* For MiniGPT-4 (Vicuna), set the LLM path 
[here](minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) at Line 18

### miniGPT4v 下载 & 修改 .yaml


Download the pretrained model checkpoints


| MiniGPT-v2 (after stage-2) | MiniGPT-v2 (after stage-3) | MiniGPT-v2 (online developing demo)| 
|------------------------------|------------------------------|------------------------------|
| [Download](https://drive.google.com/file/d/1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu/view?usp=sharing) |[Download](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view?usp=sharing) | [Download](https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view?usp=sharing) |


For **MiniGPT-v2**, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigptv2_eval.yaml](eval_configs/minigptv2_eval.yaml#L10) at Line 8.



| MiniGPT-4 (Vicuna 13B) | MiniGPT-4 (Vicuna 7B) | MiniGPT-4 (LLaMA-2 Chat 7B) |
|----------------------------|---------------------------|---------------------------------|
| [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) | [Download](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing) |

For **MiniGPT-4**, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10) at Line 8 for Vicuna version or [eval_configs/minigpt4_llama2_eval.yaml](eval_configs/minigpt4_llama2_eval.yaml#L10) for LLama2 version.   

# 运行

### 使用 Gradio 交互界面：


For MiniGPT-v2, run
```python
python demo_v2.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0
```

For MiniGPT-4 (Vicuna version), run

```python
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

For MiniGPT-4 (Llama2 version), run

```python
python demo.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0
```

### 命令行运行（不使用使用 Gradio 交互界面）：



```python
python demo_no_gradio.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

由于服务器无法现实图形界面，因此简单改写得到 demo_no_gradio.py：