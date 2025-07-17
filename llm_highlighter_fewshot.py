import ast
import json
import logging
import random
from itertools import product

import numpy as np
import pulp
from openai import OpenAI
from sklearn.metrics import average_precision_score, f1_score

from fin_rag.train_diff_agg import compute_metrics

logger = logging.getLogger(__name__)

random.seed(42) 

def pad_and_stack(predictions_list, labels_list, pad_label=-100):
    """
    Given a list of predictions and labels arrays (each with shape (1, seq_len, ...)),
    pad each to the maximum sequence length along the sequence dimension and then stack.

    predictions_list: list of numpy arrays of shape (1, seq_len, num_labels)
    labels_list: list of numpy arrays of shape (1, seq_len)
    pad_label: the value with which to pad the labels (e.g. -100)

    Returns:
        final_preds: numpy array of shape (n_examples, max_seq_len, num_labels)
        final_labels: numpy array of shape (n_examples, max_seq_len)
    """
    # Determine maximum sequence length across all examples.
    max_seq_len = max(arr.shape[1] for arr in predictions_list)

    padded_preds = []
    padded_labels = []
    for pred, lab in zip(predictions_list, labels_list):
        # pred has shape (1, seq_len, num_labels) and lab has shape (1, seq_len)
        seq_len = pred.shape[1]
        pad_len = max_seq_len - seq_len

        # Pad predictions with zeros; padding only on the sequence dimension.
        if pad_len > 0:
            padded_pred = np.pad(pred, ((0,0), (0, pad_len), (0,0)), mode='constant', constant_values=0)
            padded_lab = np.pad(lab, ((0,0), (0, pad_len)), mode='constant', constant_values=pad_label)
        else:
            padded_pred = pred
            padded_lab = lab

        padded_preds.append(padded_pred)
        padded_labels.append(padded_lab)

    # Concatenate along the batch (first) axis.
    final_preds = np.concatenate(padded_preds, axis=0)
    final_labels = np.concatenate(padded_labels, axis=0)
    return final_preds, final_labels

def main():
    
    client = OpenAI()

    training_annotated_results = []
    with open('fin_rag/annotation/annotated_result/all/setting2/train.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            id_text_label = [
                data['sample_id'],
                data['text'],
                data['tokens'],
                data['naive_aggregation']['label'],
                data['highlight_probs']
            ]
            training_annotated_results.append(id_text_label)

    expert_annotated_results = []
    with open('fin_rag/annotation/annotated_result/all/setting2/expert.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            key = list(data.keys())[0]
            id_text_label = [
                key,
                data[key]['text'],
                data[key]['tokens'],
                data[key]['binary_labels']
            ]
            expert_annotated_results.append(id_text_label)


    predictions_list = []
    labels_list = []
    for expert_element in expert_annotated_results:
        random_indices = random.sample(range(len(training_annotated_results)), 10)
        filtered_list1 = [string for string, flag in zip(training_annotated_results[random_indices[0]][2], training_annotated_results[random_indices[0]][3]) if flag == 1]
        filtered_list2 = [string for string, flag in zip(training_annotated_results[random_indices[1]][2], training_annotated_results[random_indices[1]][3]) if flag == 1]
        filtered_list3 = [string for string, flag in zip(training_annotated_results[random_indices[2]][2], training_annotated_results[random_indices[2]][3]) if flag == 1]
        filtered_list4 = [string for string, flag in zip(training_annotated_results[random_indices[3]][2], training_annotated_results[random_indices[3]][3]) if flag == 1]
        filtered_list5 = [string for string, flag in zip(training_annotated_results[random_indices[4]][2], training_annotated_results[random_indices[4]][3]) if flag == 1]
        filtered_list6 = [string for string, flag in zip(training_annotated_results[random_indices[5]][2], training_annotated_results[random_indices[5]][3]) if flag == 1]
        filtered_list7 = [string for string, flag in zip(training_annotated_results[random_indices[6]][2], training_annotated_results[random_indices[6]][3]) if flag == 1]
        filtered_list8 = [string for string, flag in zip(training_annotated_results[random_indices[7]][2], training_annotated_results[random_indices[7]][3]) if flag == 1]
        filtered_list9 = [string for string, flag in zip(training_annotated_results[random_indices[8]][2], training_annotated_results[random_indices[8]][3]) if flag == 1]
        filtered_list10 = [string for string, flag in zip(training_annotated_results[random_indices[9]][2], training_annotated_results[random_indices[9]][3]) if flag == 1]

        ls = expert_element[2]
        query_string = (
            "Analyze the significance of tokens from a financial report paragraph and return only "
            "the most important ones in their original order. Ensure that each output token exactly "
            "matches one from the input list, including punctuation. Provide no additional text, explanation, or formatting.\n\n"
            "Example 1:\n"
            f"Input: {training_annotated_results[random_indices[0]][2]}\n"
            f"Output: {filtered_list1}\n\n"
            "Example 2:\n"
            f"Input: {training_annotated_results[random_indices[1]][2]}\n"
            f"Output: {filtered_list2}\n\n"
            "Example 3:\n"
            f"Input: {training_annotated_results[random_indices[2]][2]}\n"
            f"Output: {filtered_list3}\n\n"
            "Example 4:\n"
            f"Input: {training_annotated_results[random_indices[3]][2]}\n"
            f"Output: {filtered_list4}\n\n"
            "Example 5:\n"
            f"Input: {training_annotated_results[random_indices[4]][2]}\n"
            f"Output: {filtered_list5}\n\n"
            "Example 6:\n"
            f"Input: {training_annotated_results[random_indices[5]][2]}\n"
            f"Output: {filtered_list6}\n\n"
            "Example 7:\n"
            f"Input: {training_annotated_results[random_indices[6]][2]}\n"
            f"Output: {filtered_list7}\n\n"
            "Example 8:\n"
            f"Input: {training_annotated_results[random_indices[7]][2]}\n"
            f"Output: {filtered_list8}\n\n"
            "Example 9:\n"
            f"Input: {training_annotated_results[random_indices[8]][2]}\n"
            f"Output: {filtered_list9}\n\n"
            "Example 10:\n"
            f"Input: {training_annotated_results[random_indices[9]][2]}\n"
            f"Output: {filtered_list10}\n\n"
            f"The input I want you to extract keywords from:\n"
            f"{ls}"
        )

        
        # Retry loop until valid output is received
        while True:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query_string}
                ]
            )
            output = completion.choices[0].message.content
            
            try:
                ls2 = ast.literal_eval(output)
                break  # Exit the loop if parsing succeeds
            except Exception as e:
                logger.warning("Parsing failed, retrying: %s", e)

        result = [1 if item in ls2 else 0 for item in ls]

        pred_np = np.zeros((1, len(result), 2), dtype=float)
        for i, r in enumerate(result):
            pred_np[0, i, int(r)] = 1.0

        lab_np  = np.array(expert_element[3], dtype=int)[None, :]

        predictions_list.append(pred_np)
        labels_list.append(lab_np)
    
    predictions, labels = pad_and_stack(predictions_list, labels_list, pad_label=-100)
    expert_metrics = compute_metrics((predictions, labels))
    logger.info("Expert Metrics: %s", expert_metrics)
    


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()
