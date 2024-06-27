# Build NLQA Using LLMs
This repository contains the implementation of a Natural Language Question-Answering (NLQA) system based on the GPT-neo model and huggingface transformers library. The base GPT-neo 1.3 B model is finetuned on Stanford CS324 LLM lecture notes data webscraped and collected through a python script. Parameter Efficient Fine Tuning (PEFT) method used to create and add Low Rank Adapter (LoRA) to the model.

## Table of Contents
- Overview
- Features
- Results

## Overview 
Initially Anthropic-claude / ChatOpenAI model was chosen , but due to these models being closed source and paid , GPT-neo was chosen. ollama was also not used as size to locally download the models was too high. Llama-3(8B) and Llama-2(7B) were also too large inorder to fit in the Kaggle provided GPUs. Even the GPT-NEO 1.3B and 125m version was chosen instead of 2.7B due to size constraints. The text was first scraped through the webscraper.py script. Through the main lectures link, all the text present on the sublinks present was collected and stored in the data0.txt file. Text was then cleaned and tokenized into chunks of 96 size (Due to memory constraints , a higher size was not possible) and then encoded through the model tokenizer and converted into dictionary with input_ids and attention_mask. A custom dataset was created inorder to properly pass these values to the model and an extra  field 'labels' was created for supervised learning. Initially , a custom pytorch training loop was used inorder to maximise the training while following the memory constraints. Mixed precision , Distributed parallel data processing on multiple gpus , gradient clipping methods were employed. At the start , the entire model was trained on the data, and then optimal number of freezing of layers was experimented. At the end, the model was trained through huggingface transformers trainer object with the smaller version and using PEFT LoRA adapter. This provided the best results.

## Features
- GPT-Neo Fine-Tuning: Utilizes GPT-Neo models for natural language understanding and question answering.
- PEFT with LoRA Adapters: Fine-tunes specific layers to efficiently incorporate new knowledge without degrading overall model performance.
- Data Management: Employs sentence embeddings, attention masks, and ChromaDB for efficient data handling.
- Custom PyTorch Training Loop: Includes gradient clipping, mixed precision, and garbage cleaning to optimize memory usage.
- Web Scraping Bot: Processes and cleans text data for training.

## Results
 The model was trained for 7 epochs: 
 ![Training Summary Image](https://github.com/eshan1347/GPT-NEO-LORA/blob/main/Screenshot%20from%202024-06-27%2015-29-26.png)

 Difference between original model(below) and fine tuned one (above):
 ![Difference between outputs of the two models](https://github.com/eshan1347/GPT-NEO-LORA/blob/main/Screenshot%20from%202024-06-27%2015-42-27.png)

 
