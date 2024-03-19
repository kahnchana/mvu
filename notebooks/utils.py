import urllib.request
import json

import torch


DATA_PATHS = {
    "ego_schema_q" : "https://github.com/kahnchana/mvu/releases/download/v1/questions.json",
    "ego_schema_subset_a" : "https://github.com/kahnchana/mvu/releases/download/v1/subset_answers.json",
}

def load_json_from_web(url):
    """Loads JSON data from a given URL

    Args:
        url (str): The URL to load JSON data from

    Returns:
        dict: The parsed JSON data as a dictionary
    """
    response = urllib.request.urlopen(url)
    data = response.read()

    # Decode the bytes to string
    json_data = data.decode('utf-8')

    # Parse the JSON
    parsed_json = json.loads(json_data)

    return parsed_json


def calc_loglikelihood(logits, labels):
    """
    Calculates the loglikelihood of the model's predictions given the labels.

    Args:
        logits (torch.Tensor): The model's predictions with shape
            (batch_size, sequence_length, num_classes)
        labels (torch.Tensor): The labels with shape
            (batch_size, sequence_length)

    Returns:
        torch.Tensor: The calculated loglikelihood with shape (batch_size,)
    """
    # First, we need to remove the last token from logits since it does
    # not have a corresponding label. Also, we need to move the labels
    # to the same device as logits.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.to(shift_logits.device)

    # Now, we can calculate the loglikelihood. We use cross-entropy loss
    # with reduction set to 'none' to get the loss for each element in
    # the batch. We then view the loss as a 2D tensor where each row
    # corresponds to a batch element and each column corresponds to a
    # timestep. We sum the losses across all timesteps and divide by
    # the number of valid labels in each batch element.
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    loglikelihood = loss_func(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    loglikelihood = loglikelihood.view(shift_logits.size(0), -1).sum(-1)
    loglikelihood = loglikelihood / (shift_labels != -100).sum(-1)

    return loglikelihood


def get_ego_schema(opts=None):
    """
    Get the EgoSchema dataset.

    The dataset is loaded from two JSON files, one containing questions
    and another containing answers. The answers are filtered to only
    contain those for questions that have answers.

    Each element in the returned list is a dict with the following keys:

    - q_uid: The question id
    - ans: The answer
    - question: The question text
    - prompts: A list of prompts, each of which is the question followed
      by one of the answer options
    - only_prompts: A list of just the answer options

    Args:
        opts: Unused argument

    Returns:
        A list of dicts containing the dataset
    """
    question_data = load_json_from_web(DATA_PATHS['ego_schema_q'])
    answer_data = load_json_from_web(DATA_PATHS['ego_schema_subset_a'])

    question_data_filtered = [
        {**x, 'ans': answer_data[x["q_uid"]]}
        for x in question_data
        if x["q_uid"] in answer_data.keys()
    ]
    dataset = []

    for datum in question_data_filtered:
        cur = {
            'q_uid': datum['q_uid'],
            'ans': datum['ans'],
            'question': datum['question'],
            'prompts': [
                f"{datum['question']} {datum[f'option {x}']}"
                for x in range(5)
            ],
            'only_prompts': [f"{datum[f'option {x}']}" for x in range(5)]
        }
        dataset.append(cur)
    
    return dataset
