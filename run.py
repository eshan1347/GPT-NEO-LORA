from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, GPTNeoConfig
import torch
from peft import get_peft_config, PeftType, LoraConfig ,get_peft_model,PeftModel, PeftConfig

device = 'cpu'

# Load tokenizer from a pre-trained model
# tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer = GPT2Tokenizer.from_pretrained('./out/kaggle/working/models')
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer Loaded")

# Load model from a pre-trained model
# model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
model = GPTNeoForCausalLM.from_pretrained('./out/kaggle/working/models')
print("Model Loaded")

# save_directory = './out/kaggle/working/models'
# model.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)
print("Model and Tokenizer Saved")


peft_model_path = './gptNeo13_lora/kaggle/working/fine_tuned_model'
model = PeftModel.from_pretrained(model, peft_model_path).to(device)
print('Peft Lora Adapter loaded')

def generate_answer(question, max_length=100):
    input_text = f"Question: {question}\nAnswer:"
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(question, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # return answer.split("Answer:")[-1].strip()
    return answer

print('Enter prompt: ')
print(generate_answer(input()))



# model1.to(device).eval()
# #ip = preproTxt(input())
# tokenizer1.pad_token = tokenizer1.eos_token
# ip = tokenizer1(input(),padding=True,truncation=True, return_tensors='pt')
# ip.to(device)

# with torch.no_grad():
#     output_ids = model1.generate(
#         input_ids=ip['input_ids'],
#         attention_mask=ip['attention_mask'],
#         max_length=500,  # Adjust the max_length as needed
# #         num_return_sequences=1,
# #         no_repeat_ngram_size=2,  # Optional: to reduce repetition
#         temperature=0.7,  # Optional: controls the creativity
#         top_k=50,  # Optional: top-k sampling
#         top_p=0.95,  # Optional: top-p sampling (nucleus sampling)
#         do_sample=True  # Set to True for sampling, False for greedy decoding
#     )

# # Decode the output
# generated_text = tokenizer1.decode(output_ids[0], skip_special_tokens=True)

# print(generated_text)
