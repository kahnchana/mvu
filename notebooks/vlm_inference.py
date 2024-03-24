import os
from PIL import Image
from tqdm import tqdm

import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from utils import get_ego_schema, calc_loglikelihood, download_ego_schema_center_frames


def prepare_inputs(image, datum, model_config, tokenizer, image_processor):

    image_tensor = process_images([image], image_processor, model_config)[0]

    qs = datum['question']
    prompt_list = datum['only_prompts']

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


def eval_model(args):
    
    # Load Model
    disable_torch_init()
    model_path = os.path.expanduser("liuhaotian/llava-v1.5-13b")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name)

    # Setup conversation mode
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # Load Data
    dataset = get_ego_schema()
    data_root = download_ego_schema_center_frames(save_path="temp_data/")

    correct, total = 0, 0
    for index, datum in tqdm(enumerate(dataset), total=len(dataset)):
        # load center frame
        frame_path = f"{data_root}/{datum['q_uid']}.mp4"
        c_frame = Image.open(frame_path)
        
        batch, raw_prompts = prepare_inputs(c_frame, datum, model.config, tokenizer, image_processor)
        batch = {x: y.to(device='cuda', non_blocking=True) for x, y in batch.items()}
        batch['images'] = batch['images'].to(dtype=torch.float16)
        with torch.inference_mode():
            outputs = model(**batch)

        seq_len = batch['labels'].shape[-1]
        loss = calc_loglikelihood(outputs.logits.detach()[:, -seq_len:], batch['labels'])
        pred = loss.argmin().item()

        answer = datum['ans']
        correct += answer == pred
        total += 1
        if (total + 1) % 100 == 0:
            print(f"Accuracy: {correct / total}")

    print(f"Final Accuracy: {100 * correct / total} %")
