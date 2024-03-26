import argparse
import json

import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import (OwlViTForObjectDetection, OwlViTProcessor)

decord.bridge.set_bridge('torch')

def draw_image(vis_image, save_path, box_list, label_list):
    """
    Draw boxes and labels on an image and save the modified image to a specified path.

    Args:
        vis_image (PIL.Image): The original image to be modified.
        save_path (str): The file path where the modified image will be saved.
        box_list (list): List of boxes with coordinates.
        label_list (list): List of labels for the boxes.

    Returns:
        None
    """
    # Create a drawing object
    draw = ImageDraw.Draw(vis_image)

    # Draw each box on the image
    for box, label in zip(box_list, label_list):
        draw.rectangle(box, outline='red', width=3)
        draw.text((box[0], box[1]), label, fill='red', font_size=12)

    # Save the modified image
    vis_image.save(save_path)


def get_ego_scheme(data_root, no_answer=False):
    """
    Retrieves a dataset from the given data root, optionally filtering out questions with no answers (test set).
    
    Args:
        data_root (str): The root directory containing the questions and answers data.
        no_answer (bool, optional): Whether to exclude questions with no answers. Defaults to False.
    
    Returns:
        list: A list of dictionaries representing the dataset, each containing the question ID, answer, question text, prompts, only prompts, video categories, and frame index.
    """
    question_data = json.load(open(f"{data_root}/questions.json", "r"))
    answer_data = json.load(open(f"{data_root}/subset_answers.json", "r"))
    frame_idx = json.load(open(f"/home/kanchana/repo/mlvu/vlm/output/v1_labelB/video_select_idx.json", "r"))
    video_categories = json.load(open(f"/home/kanchana/repo/mlvu/vlm/output/v1_labelB/per_vid_category.json", "r"))
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
            'video_categories': video_categories[datum['q_uid']], 
            'frame_idx': int(frame_idx[datum['q_uid']])
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


def generate_data(args):

    no_answer = False
    if args.dataset.lower() == "EgoSchema".lower():
        data_root = "/raid/datasets/EgoSchema_Data"
        dataset = get_ego_scheme(data_root, no_answer=no_answer)
    elif args.dataset.lower() == "NextQA".lower():
        data_root = "/raid/datasets/nextqa"
        dataset = get_next_qa(data_root=data_root, opts=args)
        
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", device_map="auto")

    save_dict = {}
    for index, datum in tqdm(enumerate(dataset), total=len(dataset)):
        if index == 500:
            break
        
        if args.dataset.lower() == "EgoSchema".lower():
            vid_path = f"{data_root}/videos/{datum['q_uid']}.mp4"
            frames = VideoReader(vid_path, ctx=cpu(0))
        elif args.dataset.lower() == "NextQA".lower():
            vid_path = f"{data_root}/NExTVideo/{datum['video']}.mp4"
            frames = VideoReader(vid_path, ctx=cpu(0))
        elif args.dataset.lower() == "ssv2".lower():
            frames, label, label_idx, aux = datum
            datum = {"q_uid": index}

        frame_idx_list = np.linspace(0, len(frames) - 1, 8).astype(int).tolist()
        for c_frame_idx in frame_idx_list:
            c_frame = frames[c_frame_idx].numpy()
            c_frame = Image.fromarray(c_frame)

            if args.dataset.lower() == "ssv2".lower():
                orig_texts = aux['placeholders']
                texts = [f"photo of a {x}" for x in orig_texts]
                texts = [texts, ]
            else:
                orig_texts = list(set(datum['video_categories']))
                texts = [f"photo of a {x}" for x in orig_texts]
                texts = [texts, ]
            
            inputs = processor(text=texts, images=c_frame, return_tensors="pt")
            inputs = {x:y.to("cuda") for x, y in inputs.items()}
            outputs = model(**inputs)

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([c_frame.size[::-1]]).to("cuda")  # x, y = 480, 360 
            # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.5)
            if results[0]['boxes'].shape[0] == 0:
                results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
            if results[0]['boxes'].shape[0] == 0:
                results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.05)
            
            if args.save is not None:
                boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
                if datum['q_uid'] not in save_dict:
                    save_dict[datum['q_uid']] = {}    
                save_dict[datum['q_uid']][f"frame_{c_frame_idx}"] = {
                    "boxes": boxes.tolist(),
                    "scores": scores.tolist(),
                    "labels": labels.tolist(),
                    "text": orig_texts,
                }
                if (index + 1) % 10 == 0:
                    with open(args.save, 'w') as f:
                        json.dump(save_dict, f)
        
    if args.save is not None:
        with open(args.save, 'w') as f:
            json.dump(save_dict, f)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="EgoSchema")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    generate_data(args)