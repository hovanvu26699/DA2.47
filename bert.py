from transformers import BertForQuestionAnswering, AutoTokenizer
import torch
import sys
from utils import *

def question_answer(model, tokenizer, question, text, device):

    #tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text, max_length=512, truncation=True)

    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #number of tokens in segment A (question)
    num_seg_a = sep_idx+1
    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a

    #list of 0s and 1s for segment embeddings
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)

    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    answer = ""
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    return answer


print("Loading model..")
tokenizer = AutoTokenizer.from_pretrained("./data/results/qa/bert")
model = BertForQuestionAnswering.from_pretrained("./data/results/qa/bert")

with torch.no_grad():
    context =  read_docs('./data/docs.txt')

    while True:
        print("Insert the question...")
        question: str = input()
        outputs = ""
        output = question_answer(model, tokenizer, question, context,"cpu")
        print(f"Answer: {output}")
