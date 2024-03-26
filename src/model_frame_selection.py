import argparse
import json
import math
import os

import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from tqdm import tqdm

decord.bridge.set_bridge('torch')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def calc_loglikelihood(logits, labels):
    """
    Calculates the loglikelihood of the model's predictions given the labels
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    shift_labels = shift_labels.to(shift_logits.device)
    loglikelihood = loss_func(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loglikelihood = loglikelihood.view(shift_logits.size(
        0), -1).sum(-1) / (shift_labels != -100).sum(-1)
    return loglikelihood


def prepare_inputs(image, datum, model_config, tokenizer, image_processor):

    image_tensor = process_images(image, image_processor, model_config)

    prompt_list = datum['only_prompts']
    joint_question = f"{datum['question']} Options: " + "".join([f"\n{index}: {value}" for index, value in enumerate(prompt_list)])
    qs = f"How useful is this frame to answer the following question. {joint_question}"
    prompt = "This frame is highly useful and contains relevant information."
    
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # generate inputs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], prompt)
    prompt = conv.get_prompt()
    raw_prompts = prompt

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    # generate targets
    target = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    total_len = int(target.ne(tokenizer.pad_token_id).sum())
    rounds = prompt.split(conv.sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX
    for i, rou in enumerate(rounds):
        if rou == "":
            break

        parts = rou.split(sep)
        if len(parts) != 2:
            break
        parts[0] += sep

        round_len = len(tokenizer_image_token(rou, tokenizer))
        instruction_len = len(
            tokenizer_image_token(parts[0], tokenizer)) - 2

        target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

        cur_len += round_len
    target[cur_len:] = IGNORE_INDEX

    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" (ignored)"
            )
    labels = target

    # batch together 5 prompts
    input_ids = input_ids.unsqueeze(0).repeat(8, 1)
    labels = labels.unsqueeze(0).repeat(8, 1)
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    batch = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        images=image_tensor,
    )
    return batch, raw_prompts


def get_ego_scheme(data_root, no_answer=True, opts=None):
    question_data = json.load(open(f"{data_root}/questions.json", "r"))
    answer_data = json.load(open(f"{data_root}/subset_answers.json", "r"))
    
    video_categories, frame_idx = None, None
    spatial, motion = False, False
    if opts is not None:
        if opts.load_global:
            video_categories = json.load(open(f"/home/kanchana/repo/mlvu/vlm/output/v1_labelB/per_vid_category.json", "r"))
        if opts.frame_sel:
            frame_idx = json.load(open(f"/home/kanchana/repo/mlvu/vlm/output/v1_labelB/video_select_idx.json", "r"))
        if opts.load_spatial:
            video_categories = json.load(open(f"/home/kanchana/repo/mlvu/vlm/output/v1_labelB/spatial_v1.json", "r"))
            spatial = True
        if opts.load_motion:
            video_categories = json.load(open("/home/kanchana/repo/mlvu/vlm/output/v1_labelB/motion_v1.json", "r"))
            motion = True

    if no_answer:
        question_data_filtered = question_data
    else:
        question_data_filtered = [{**x, 'ans': answer_data[x["q_uid"]]}
                                  for x in question_data if x["q_uid"] in answer_data.keys()]
    dataset = []
    for datum in question_data_filtered:
        cur = {
            'q_uid': datum['q_uid'],
            'ans': None if no_answer else datum['ans'],
            'question': datum['question'],
            'prompts': [f"{datum['question']} {datum[f'option {x}']}" for x in range(5)],
            'only_prompts': [f"{datum[f'option {x}']}" for x in range(5)],
            'video_categories': video_categories[datum['q_uid']] if video_categories is not None else None, 
            'frame_idx': int(frame_idx[datum['q_uid']]) if frame_idx is not None else None, 
            'spatial': spatial,
            'motion': motion
        }
        dataset.append(cur)
    return dataset


def get_next_qa(data_root, opts=None):
    file_name = "anno/val.csv"
    # file_name = "test-data-nextqa/test.csv"
    remap_name = "map_vid_vidorID.json"
    remap = json.load(open(f"{data_root}/{remap_name}", "r"))
    with open(f"{data_root}/{file_name}", "r") as fo:
        question_data = fo.readlines()
    question_data = [x.strip().split(",") for x in question_data]
    header = question_data[0]
    question_data = question_data[1:]
    question_data = [{header[i]: x[i]
                      for i in range(len(header))} for x in question_data]
    
    video_categories = None
    if opts is not None:
        if opts.load_global:
            video_categories = json.load(open(f"/home/kanchana/repo/mlvu/vlm/output/v1_labelB/nextqa_val.json", "r"))
            # frame_idx = json.load(open(f"/home/kanchana/repo/mlvu/vlm/output/v1_labelB/video_select_idx.json", "r"))

    dataset = []
    for datum in question_data:
        cur = {
            'q_uid': datum['qid'],
            'video': remap[datum['video']],
            'ans': int(datum['answer']),
            'question': datum['question'] + "?",
            'prompts': [f"{datum['question']}? {datum[f'a{x}']}" for x in range(5)],
            'only_prompts': [f"{datum[f'a{x}']}" for x in range(5)],
            'video_categories': video_categories[datum['qid']] if video_categories is not None else None,
            'frame_idx': None
        }
        dataset.append(cur)

    return dataset


def eval_model(args):
    # Load Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name)

    # Setup conversation mode
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # Load Data
    no_answer = args.save is not None
    no_answer = False
    if args.dataset.lower() == "EgoSchema".lower():
        data_root = "/raid/datasets/EgoSchema_Data"
        dataset = get_ego_scheme(data_root, no_answer=no_answer, opts=args)
    elif args.dataset.lower() == "NextQA".lower():
        data_root = "/raid/datasets/nextqa"
        dataset = get_next_qa(data_root=data_root, opts=args)

    save_dict = {}
    for index, datum in tqdm(enumerate(dataset), total=len(dataset)):
        if args.dataset.lower() == "EgoSchema".lower():
            vid_path = f"{data_root}/videos/{datum['q_uid']}.mp4"
        elif args.dataset.lower() == "NextQA".lower():
            vid_path = f"{data_root}/NExTVideo/{datum['video']}.mp4"
        frames = VideoReader(vid_path, ctx=cpu(0))
        frame_idx_list = np.linspace(0, len(frames) - 1, 8).astype(int).tolist()
        frames = [Image.fromarray(frames[x].numpy()) for x in frame_idx_list]        
        batch, raw_prompts = prepare_inputs(frames, datum, model.config, tokenizer, image_processor)
        batch = {x: y.to(device='cuda', non_blocking=True)  for x, y in batch.items()}
        batch['images'] = batch['images'].to(dtype=torch.float16)
        with torch.inference_mode():
            batch_0 = {k: v[:4] for k, v in batch.items()}
            batch_1 = {k: v[4:] for k, v in batch.items()}
            outputs_0 = model(**batch_0)
            outputs_1 = model(**batch_1)
        seq_len = batch['labels'].shape[-1]
        logits = torch.cat([outputs_0.logits.detach().cpu(), outputs_1.logits.detach().cpu()], dim=0)
        loss = calc_loglikelihood(logits[:, -seq_len:], batch['labels'])
        pred = loss.argmin().item()
        save_dict[datum['q_uid']] = pred

        if index % 50 == 0:
            json.dump(save_dict, open(args.save, "w"), indent=4)
        
    json.dump(save_dict, open(args.save, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--output-folder", type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save", type=str, default="output/frame_selection_v1.json")
    parser.add_argument("--dataset", type=str, default="EgoSchema")
    parser.add_argument("--load-global", action="store_true", default=False)
    parser.add_argument("--frame-sel", action="store_true", default=False)
    parser.add_argument("--load-spatial", action="store_true", default=False)
    parser.add_argument("--load-motion", action="store_true", default=False)
    args = parser.parse_args()

    eval_model(args)
