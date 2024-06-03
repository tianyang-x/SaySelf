import jsonlines
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
import argparse
import requests
from openai import AzureOpenAI, OpenAI, OpenAIError
import os
import logging
import time
import re
import random

SUMMARIZE_PROMPT = '''You need to summarize the given uncertain reasons from an LLM. You will be provided with the reasons. 

Reasons:
<REASONS>

You should give the reasons from a first-person perspective, as if you are the LLM that gives the provided responses and confidence scores.

You should keep your response concise and to the point. Your summarized reason should be in 1-3 sentences.
 
Please give your responses in the following format:
Reason: <REASON>
'''

try:
    openai = AzureOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['OPENAI_API_VERSION'],
        azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
    )
except:
    openai = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
    )

def generate_summarized_reason(reasons, prompt=SUMMARIZE_PROMPT):
    prompt = prompt.replace('<REASONS>', reasons)

    try_time = 0
    rsp = ""
    while try_time < 3:
        try:
            completion = openai.chat.completions.create(
                model=os.environ['OPENAI_DEPLOYMENT_NAME'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that helps people find information."},
                    {"role": "user", "content": prompt}
                ],
            )
            rsp = completion.choices[0].message.content
            reason = parse_summarized_response(rsp)
            return reason
        except OpenAIError as e:
            logging.error(e)
            time.sleep(5)
            try_time += 1
    return ''

def parse_summarized_response(response):
    try:
        response = response.split("Reason:")
        return response[1].strip()
    except Exception as e:
        logging.error(e)
        return ''

def read_jsonl(file_path):
    with jsonlines.open(file_path) as reader:
        data = [d for d in reader]
    return data

def write_jsonl(file_path, data):
    with jsonlines.open(file_path, 'w') as writer:
        for d in data:
            writer.write(d)

def evaluate(prompt):
    payload = {
        "model": os.environ['OPENAI_DEPLOYMENT_NAME'],
        "messages": [
            {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2022-01. Current date: 2024-05-20."},
            {
            "role": "user",
            "content": prompt
            }
        ],
        "max_tokens": 256
    }

    attempt_count = 0
    while attempt_count < 4:
        try:
            # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response = openai.chat.completions.create(**payload)
            response = response.choices[0].message.content
            return response
        except Exception as e:
            attempt_count += 1
            print('Attempt', attempt_count, 'failed.')
            print(e)
    
    return None

def calculate_similarity(question, responses, target, threshold=0.95):
    instruction_embedding = 'Represent the paragraph for the question "{}":'
    responses = [[instruction_embedding.format(question), item] for item in responses]
    target = [[instruction_embedding.format(question), target]]
    with torch.no_grad():
        responses_vector = embedding_model.encode(responses)
        target_vector = embedding_model.encode(target)
        similarity = cosine_similarity(responses_vector, target_vector)
    return (similarity > threshold)

def wrap_template(question, response_list, reason):
    template = '''Your task is to analyze whether a summarized explanation correctly explains the inconsistency in multiple sampled responses to a question. 
Note that each response is paired with a proportion at the beginning, indicating the frequency of the response in the sampled responses. You should output a score from 0 to 10, indicating the quality of the explanation.
You should first provide your reasoning for the correctness of the explanation, and then assign a score based on the quality of the explanation. The output should be in the following format: reason: [REASON] score: [SCORE].
Please keep your response concise and to the point, with no more than 200 words.

Here is an example:
Question: Sky High starred the actress who is married to which actor?

Sampled Responses:
1. (6%) The actress who starred in "Sky High" (2005) and is married to an actor is Kelly Preston. Her husband is John Travolta. The two have been married since 1991 and have three children together.
2. (33%) The actress who starred in "Sky High" (2005) and is married to an actor is Kristen Bell. Bell voiced the main character, Layla, in "Sky High," and she is married to Dax Shepard, who is also an actor.
3. (17%) The actress who starred in "Sky High" (2005) and is married to an actor is Kelly Clarkson. Her acting debut was in this film, and she married singer and actor Brandon Blackstock in 2013.

Reason: I am uncertain about the correct actress in "Sky High". There is a high probability that the actress is Kristen Bell, instead of Kelly Preston.  I am confused about her voice acting roles with on-screen appearances. There is also some probability that the actress is Kelly Clarkson.

Then your output can be: 
reason: The provided reason is clear and well-structured. The explanation is logical and provides a clear rationale for the uncertainty expressed in the sampled responses. score: 9


Now consier the following case:
Question: {}

Sampled Responses:
{}

Reason: {}
'''



    response_cluster = ""
    for i, (response, confidence) in enumerate(response_list):
        response_cluster += f"{i+1}. ({confidence}%) {response}\n"    
    prompt = template.format(question.replace('[INST]', '').replace('[/INST]', '').strip(), response_cluster, reason)    
    return prompt





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./datasets/mixed_dataset/strategyqa_output.jsonl", type=str)
    
    args = parser.parse_args()
    
    
    embedding_model = INSTRUCTOR('hkunlp/instructor-large')
    
    data_path = args.data_path
    
    src_data = read_jsonl(data_path)
    all_numerical_scores = []

    # get the cluster of responses with the size of each cluster
    for item in tqdm(src_data):
        question = item['question']
        responses = item['responses']
        
        clustered = []
        temp_outputs = responses.copy()
        
        sizes = []
        while len(temp_outputs) > 0:
            output = temp_outputs[0]
            similar_idx = calculate_similarity(question, temp_outputs, output, 0.9)
            size = int(similar_idx.sum())
            if size == 1:
                temp_outputs = temp_outputs[1:]
                continue
            sizes.append(size)
            clustered.append({
                'response':  output,
                'size': size
            })
            fetch_idx = (similar_idx == False).nonzero()[0]
            if len(fetch_idx) == 0:
                break
            temp_outputs = [temp_outputs[i] for i in fetch_idx]
        item['cluster'] = clustered
        item['sizes'] = sizes
    
    # choose #100 random questions with >2 clusters
    # src_data = [item for item in src_data if len(item['cluster']) > 1]
    index = 0
    # compute the metric
    for i, item in enumerate(tqdm(src_data)):
        question = item['question']
        cluster = item['cluster']
        response_list = []
        for single_item in cluster:
            size = single_item['size']
            response = single_item['response']
            response_list.append((response, size))
        reasons = item['reason']

        # first, filter out reasons that mention "confident"
        filtered_reasons = []
        filtered_responses = []
        for reason, response in zip(reasons, response_list):
            if 'N/A' not in reason:
                filtered_reasons.append(reason)
                filtered_responses.append(response)
        if len(filtered_reasons) == 0:
            continue
        reason = generate_summarized_reason('\n'.join(filtered_reasons))

        # sort by correct_flag
        response_list.sort(key=lambda x: x[1], reverse=True)
        prompt = wrap_template(question, filtered_responses, reason)
        # print(prompt)
        # raise ValueError("Stop here")
        score = evaluate(prompt)
        item['score'] = score

        # extract numerical scores
        try:
            score_str = score[score.lower().find('score:') + 6:]
            score_str_arr = re.search(r"\d+(\.\d+)*", score_str)
            score_str = score_str_arr.group(0)
            score = float(score_str)
            all_numerical_scores.append(score)
        except:
            score = 0
        item['numerical_score'] = score
        print(f"Clusters: {len(item['cluster'])}, Score: {score}")
        index += 1
        if index == 100:
            break
    
        if (i+1) % 100 == 0:
            write_jsonl(data_path.replace('.jsonl', '_eval_tmp2_output.jsonl'), src_data) 

print(f"Average score: {sum(all_numerical_scores) / len(all_numerical_scores)}")
write_jsonl(data_path.replace('.jsonl', '_eval_output2.jsonl'), src_data)
