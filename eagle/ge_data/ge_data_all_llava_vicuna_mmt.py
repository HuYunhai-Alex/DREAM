import argparse
import base64
from io import BytesIO
from tqdm import tqdm

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from fastchat.model.model_adapter import get_conversation_template
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from torch.utils.data import Dataset, DataLoader


#bigname="/home/apc/models/vicuna-7b-v1.3/"

target_model_id = "/home/apc/models/llava-v1.6-vicuna-13b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 开启 4-bit 量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用的数值类型，可选：torch.float16, torch.bfloat16 等
    bnb_4bit_quant_type="nf4",             # 量化类型，常见的选项有 "nf4" 或 "fp4"
    bnb_4bit_use_double_quant=True         # 是否使用 double quantization（一般会带来更好的精度）
)

bigmodel = LlavaNextForConditionalGeneration.from_pretrained(
    target_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
   # quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(target_model_id)

if "vicuna" in target_model_id:
    assist_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    end_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)
elif "mistral" in target_model_id:
    assist_tokens = processor.tokenizer.encode("[/INST]:", add_special_tokens=False)
    end_tokens = processor.tokenizer.encode("[INST]:", add_special_tokens=False)
    image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)

assist_len = len(assist_tokens)
end_len = len(end_tokens)

# 定义解码 base64 图像的函数
def load_image(base64_string):
    # 注意：这里假设传入的是 base64 编码的字符串
    base64_string = base64_string.replace("\n", "")
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def find_next_non_image_token(input_ids, image_token_ids):
    """
    找到 input_ids 中最后一个 image token，并返回下一个非-image token 的索引。
    
    Args:
        input_ids (torch.Tensor): 形状为 (seq_len,) 的 token ID 张量。
        image_token_ids (set): 包含所有 image token ID 的集合。

    Returns:
        int: 下一个非-image token 的索引（如果找到），否则返回 -1
    """
    # 找到所有 image token 的索引
    image_indices = torch.where(torch.tensor([id in image_token_ids for id in input_ids]))[0]
    
    if len(image_indices) == 0:
        return -1  # 没有找到 image token

    # 获取最后一个 image token 的索引
    last_image_index = image_indices[-1].item()

    # 找到下一个非-image token
    for i in range(last_image_index + 1, len(input_ids)):
        if input_ids[i].item() not in image_token_ids:
            return i  # 返回第一个非 image token 的索引

    return -1  # 没有找到下一个非-image token

# 定义自定义数据集类
class MMTBenchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 从 Hugging Face 数据集中获取单条数据（包含 "image" 和 "question"）
        record = self.dataset[idx]
        image_base64 = record["image"]
        conversation = record["conversation"]
        # 将 base64 字符串转换为 PIL.Image 对象
        image = load_image(image_base64)
        return image, conversation  # image_base64 备用（写入文件时可能需要）

# 自定义 collate 函数：批量构造 prompt 并编码
def collate_fn(batch):
    # batch 是一个列表，每个元素为 (image, question, image_base64)
    images, conversations = zip(*batch)
    print(conversations)

    # 利用 processor 构造生成 prompt
    prompts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
    # 将图像和文本传入 processor，返回编码后的模型输入
    
    return prompts, list(images)

def mid_feature_collect(hidden_states):
    hid = torch.stack(hidden_states, dim=0)
    L, B, S, D = hid.shape

    # 定义候选层范围
    candidate_start = 3
    candidate_end = int(0.75 * (L - 1))
    if candidate_end <= candidate_start:
        raise ValueError("候选层范围无效")

    # 候选层数量
    candidate_count = candidate_end - candidate_start

    # 提取候选层及其相邻层
    # 注意：候选层的索引为 candidate_start 到 candidate_end - 1
    candidate = hid[candidate_start:candidate_end, :, :, :]       # shape: (candidate_count, B, S, D)
    candidate_prev = hid[candidate_start - 1:candidate_end - 1, :, :, :]  # shape: (candidate_count, B, S, D)
    candidate_next = hid[candidate_start + 1:candidate_end + 1, :, :, :]  # shape: (candidate_count, B, S, D)

    # 全局 anchor 层：第一层和最后一层
    anchor_left = hid[0].unsqueeze(0)    # shape: (1, B, S, D)
    anchor_right = hid[-1].unsqueeze(0)   # shape: (1, B, S, D)

    def cosine_distance(x, y, eps=1e-9):
        """
        计算余弦距离：
        d(x,y) = 1 - (x · y) / (||x|| ||y|| + eps)
        输入 x, y 的形状为 (..., D)
        返回形状为 x 和 y 除最后一维外的张量
        """
        dot_xy = (x * y).sum(dim=-1)
        norm_x = x.norm(dim=-1)
        norm_y = y.norm(dim=-1)
        return 1 - dot_xy / (norm_x * norm_y + eps)

    # 计算全局距离
    # 分别计算候选层与 anchor_left 和 anchor_right 的距离
    global_left = cosine_distance(anchor_left, candidate)   # shape: (candidate_count, B, S)
    global_right = cosine_distance(anchor_right, candidate)  # shape: (candidate_count, B, S)
    global_dis = torch.abs(global_left - global_right)       # shape: (candidate_count, B, S)

    # 计算局部距离
    local_left = cosine_distance(candidate_prev, candidate)   # shape: (candidate_count, B, S)
    local_right = cosine_distance(candidate_next, candidate)  # shape: (candidate_count, B, S)
    local_dis = torch.abs(local_left - local_right)           # shape: (candidate_count, B, S)

    # 综合距离
    total_dis = global_dis + local_dis  # shape: (candidate_count, B, S)

    # 对候选层维度（dim=0）取 argmin 得到最佳候选索引 (范围在 [0, candidate_count-1])
    best_candidate_idx = torch.argmin(total_dis, dim=0)  # shape: (B, S)

    # 转换为原始 hidden_states 的层 id
    best_layer = best_candidate_idx + candidate_start  # shape: (B, S)
    print(best_layer)


    hidden_states_trans = hid.permute(1, 2, 0, 3)  # (B, S, L, D)

    # best_layer: (B, S) → 扩展为 (B, S, 1) 用于索引
    best_layer_unsq = best_layer.unsqueeze(-1)  # (B, S, 1)

    # 利用 torch.gather 在第 2 维（即 L 维度）上获取对应层的特征
    # 首先需要将 best_layer_unsq 扩展到和最后一个特征维 D 对齐，形状变为 (B, S, 1, D)
    index = best_layer_unsq.unsqueeze(-1).expand(-1, -1, -1, hidden_states_trans.shape[-1])
    # 使用 gather 得到 (B, S, 1, D) 的结果
    selected_features = torch.gather(hidden_states_trans, dim=2, index=index)

    # squeeze 掉第 2 维（长度为 1）即可得到 (B, S, D)
    selected_features = selected_features.squeeze(2)
    return selected_features

ds = load_dataset("/home/apc/Bingle/analysis", data_files="/home/apc/Bingle/mmt-bench-llava-v1.6-vicuna-7b.jsonl", split="train")
batch_size = 1
ds = ds.shuffle(seed=42)
if len(ds) < args.end:
    args.end = len(ds)
ds = ds.select(range(args.start, args.end))
dataset = MMTBenchDataset(ds)
data_loader = DataLoader(
    dataset, 
    batch_size=1,
    shuffle=True, 
    num_workers=4,        # 可根据 CPU 核数进行调整
    collate_fn=collate_fn,
    pin_memory=True
)

@torch.no_grad()
def ge(data):
    print(data)
    prompts, images = data
    # 模型生成回答
    inputs = processor(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=5120, 
        padding_side='left'
    )
    # 将数据移动到 GPU
    inputs = inputs.to("cuda:0")
    print(processor.tokenizer.decode(inputs.input_ids[0]))
    outs_big = bigmodel(**inputs, output_hidden_states=True)
    loss_mask = torch.zeros_like(inputs.input_ids)

    for i in range(inputs.input_ids.size(0)):
        tokens = inputs.input_ids[i]
        start_idx = None
        j = 0
        while j < tokens.size(0):
            # 如果还未找到起始位置，检查当前 slice 是否等于 assist_tokens
            if start_idx is None and j <= tokens.size(0) - assist_len and tokens[j:j+assist_len].tolist() == assist_tokens:
                start_idx = j  # 记录 "ASSISTANT:" 起始位置
                j += assist_len  # 跳过这段 token 序列
                continue
            # 如果已找到起始位置，检查是否匹配 end_tokens
            if start_idx is not None and j <= tokens.size(0) - end_len and tokens[j:j+end_len].tolist() == end_tokens:
                # 设置 loss mask：从 "ASSISTANT:" 序列之后到结束 token 之前
                loss_mask[i, start_idx+assist_len:j] = 1
                print(loss_mask[i], start_idx, j)
                start_idx = None  # 重置起始标记
                j += end_len  # 跳过结束 token 序列
                continue
            j += 1
            

    td={"loss_mask":loss_mask.cpu()}
    td["attention_mask"]=inputs.attention_mask.cpu()
    # early exit layer 
    # exit at layer2 for vicuna-7B and layer3 for vicuna-13B 
    td[f"inputs_embeds"] = outs_big.hidden_states[0].cpu()
    td[f"hidden_state_layer8"] = outs_big.hidden_states[8].cpu()
    td[f"hidden_state_layer16"] = outs_big.hidden_states[16].cpu()
    td[f"hidden_state_layer24"] = outs_big.hidden_states[24].cpu()
    td[f"hidden_state_mid"] = mid_feature_collect(outs_big.hidden_states).cpu()
    td[f"target"] = outs_big.hidden_states[-1].cpu()

    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for data in tqdm(data_loader):
    outdata = ge(data)
    writedata(outdir,outdata)


