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

    image_tensor = process_images([image], image_processor, model_config)[0]

    qs = datum['question']
    prompt_list = datum['only_prompts']

    if datum['video_categories'] is not None and len(datum['video_categories']):
        if datum['spatial']:
            qs = "Consider following objects located at (x, y) coordinates in video to answer the question:" + ", ".join(datum['video_categories']) + ". " + qs
        elif datum['motion']:
            qs = "Consider following objects moving along (x,y,area) trajectories in video to answer the question:" + ", ".join(datum['video_categories']) + ". " + qs
        else:
            qs = "Consider following objects in video:" + ", ".join(datum['video_categories']) + ". " + qs

    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    input_ids = []
    labels = []
    raw_prompts = []
    for prompt in prompt_list:
        # generate inputs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], prompt)
        prompt = conv.get_prompt()
        raw_prompts.append(prompt)
        cur_input_id = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        input_ids.append(cur_input_id)

        # generate targets
        target = cur_input_id.clone()
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
        labels.append(target)

    # batch together 5 prompts
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX)
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    image_tensor = image_tensor.repeat(5, 1, 1, 1)

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
            video_categories = json.load(open(f"data/output/v1_labelB/per_vid_category.json", "r"))
        if opts.frame_sel:
            frame_idx = json.load(open(f"data/output/v1_labelB/video_select_idx.json", "r"))
        if opts.load_spatial:
            video_categories = json.load(open(f"data/output/v1_labelB/spatial_v1.json", "r"))
            spatial = True
        if opts.load_motion:
            video_categories = json.load(open("data/output/v1_labelB/motion_v1.json", "r"))
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
        if opts.load_motion:
            video_categories = json.load(open(f"data/output/v1_labelB/nextqa_val.json", "r"))
            frame_idx = json.load(open(f"data/output/v1_labelB/video_select_idx.json", "r"))

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
            'frame_idx': int(frame_idx[datum['q_uid']]) if frame_idx is not None else None, 
        }
        dataset.append(cur)

    return dataset


def get_robot_dataset(opts):
    dataset = json.load(open("/raid/datasets/open_x-8frames/open_x-w-more-objects.json", "r"))
    frames =  json.load(open("data/output/v1_labelC/robot_frame_v1.json", "r"))
    motion =  json.load(open("data/output/v1_labelC/robot_motion_v1.json", "r"))

    q_template = "Taking into account all the actions performed by the robot, what can you deduce about the primary objective and focus within the video content?"

    if int(opts.split) == 0:
        dataset = dataset[:14000]
    elif int(opts.split) == 1:
        dataset = dataset[14000:]

    out_list = []
    for datum in dataset:
        name = datum['id']

        if name not in motion:
            print(f"invalid: {name}")
            motion_dict = {}
            breakpoint()
        else:
            motion_dict = motion[name]

        frame_idx = frames[name]
        selected_frame = sorted(datum['images'])[int(frame_idx)]
        prompt_list = [f"Option {index}: {value}" for index, value in enumerate(datum['choices'])]

        cur = {
            'q_uid': name,
            'ans': datum['ans_idx'],
            'question': q_template,
            'only_prompts': prompt_list,
            'video_categories': motion_dict, 
            'frame': selected_frame,
            'spatial': None,
            'motion': None
        }
        out_list.append(cur)
    
    return out_list


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
    if args.dataset.lower() == "EgoSchema".lower():
        data_root = "/raid/datasets/EgoSchema_Data"
        dataset = get_ego_scheme(data_root, no_answer=no_answer, opts=args)
    elif args.dataset.lower() == "NextQA".lower():
        data_root = "/raid/datasets/nextqa"
        dataset = get_next_qa(data_root=data_root, opts=args)
    elif args.dataset.lower() == "robot".lower():
        dataset = get_robot_dataset(args)

    correct, total = 0, 0
    save_dict = {}
    visualize = {"right": [], "wrong": []}
    for index, datum in tqdm(enumerate(dataset), total=len(dataset)):
        if args.dataset.lower() == "EgoSchema".lower():
            vid_path = f"{data_root}/videos/{datum['q_uid']}.mp4"
        elif args.dataset.lower() == "NextQA".lower():
            vid_path = f"{data_root}/NExTVideo/{datum['video']}.mp4"
        
        if args.dataset.lower() == "robot".lower():
            c_frame = Image.open(datum['frame'])
        else:
            frames = VideoReader(vid_path, ctx=cpu(0))
            c_frame_idx = int(len(frames) // 2)
            if datum['frame_idx'] is not None:
                frame_idx_list = np.linspace(0, len(frames) - 1, 8).astype(int).tolist()
                c_frame_idx = frame_idx_list[datum['frame_idx']]
            c_frame = frames[c_frame_idx].numpy()
            c_frame = Image.fromarray(c_frame)

        batch, raw_prompts = prepare_inputs(
            c_frame, datum, model.config, tokenizer, image_processor)
        batch = {x: y.to(device='cuda', non_blocking=True)
                 for x, y in batch.items()}
        batch['images'] = batch['images'].to(dtype=torch.float16)
        with torch.inference_mode():
            outputs = model(**batch)
        seq_len = batch['labels'].shape[-1]
        loss = calc_loglikelihood(outputs.logits.detach()[
                                  :, -seq_len:], batch['labels'])
        pred = loss.argmin().item()
        if no_answer:
            save_dict[datum['q_uid']] = pred

            answer = datum['ans']
            correct += answer == pred
            total += 1

            if (index + 1) % 10 == 0:
                print(f"Accuracy: {correct / total}")

            if (index + 1) % 100 == 0:
                json.dump(save_dict, open(args.save, "w"), indent=4)

        else:
            answer = datum['ans']
            correct += answer == pred

            cur_vis = {
                "question": datum['question'],
                "prompts": datum['only_prompts'],
                "answer": answer,
                "prediction": pred,
                "q_uid": datum['q_uid']
            }
            if (answer == pred):
                visualize["right"].append(cur_vis)
            else:
                visualize["wrong"].append(cur_vis)

            total += 1
            if (total + 1) % 10 == 0:
                print(f"Accuracy: {correct / total}")

    if args.save is not None:
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
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="EgoSchema")
    parser.add_argument("--load-global", action="store_true", default=False)
    parser.add_argument("--frame-sel", action="store_true", default=False)
    parser.add_argument("--load-spatial", action="store_true", default=False)
    parser.add_argument("--load-motion", action="store_true", default=False)
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
