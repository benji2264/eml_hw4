
import os
from time import time

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

# sns.set_style("darkgrid")

OUTPUT_PATH = "dl_llama2/"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
    

def task1_step2():
    # Task 1 - Step 2
    temperatures = [0.1, 0.9]
    n = 10

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for t in temperatures:
        print(f"\n\n TEMPERATURE {t}")
        for i in range(n):

            # Generate
            generate_ids = model.generate(inputs.input_ids, max_length=30, do_sample=True, temperature=t)
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f"[{i+1}] ", output)

def task1_step3():
    #     # Task 1 - Step 3
    def get_first_n_words(text, n):
        words = text.split()
        first_n_words = words[:n]
        return ' '.join(first_n_words)

    full_prompt = """In the dim glow of the setting sun, two figures sat on the edge of a weathered dock, legs dangling over the water's serene surface. The ripples below mirrored the tumultuous thoughts swirling in their minds. One broke the silence, their voice barely above a whisper, "Do you ever wonder about our existence? Why we're here, in this vast, unfathomable universe?"
# The other turned, their eyes reflecting the dying light, "All the time. It feels like we're just specks in an infinite cosmos, doesn't it? But maybe that's what makes our lives so precious. The rarity of consciousness in this vastness."
# The first nodded, "It's daunting, though. Our time is so fleeting, and yet, we spend it in pursuit of things that often don't seem to matter in the grand scale of the universe."
# "There's beauty in that pursuit, though," the second countered gently. "Perhaps the meaning isn't in the grandeur or the scale, but in the moments, the connections we forge, the love we share. Maybe our existence is defined not by the why but by the how. How we choose to live, how we impact others."
# As the sun dipped below the horizon, a comfortable silence settled between them, each lost in contemplation. The vastness of the universe, with all its mysteries and wonders, seemed a little less daunting as they sat together, ancxihored in the shared understanding that while they may not have all the answers, they had each other, and maybe, for now, that was enough."""

    n_repeat = 10
    prompt_lengths = [16, 64, 128, 256]
    
    results = {}
    t = 0.9

    # Warmp-up
    prompt = "Hey, are you conscious? Can you talk to me?"    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    for _ in range(10):
        generate_ids = model.generate(inputs.input_ids, max_length=30, do_sample=True, temperature=t)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Measure latency
    for length in prompt_lengths:
        curr_length_res = []
        print(f"Generating outputs for length {length}...")
        for _ in range(n_repeat):
            prompt = get_first_n_words(full_prompt, length)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            start = time()
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=1, do_sample=True, temperature=t)
            _ = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            curr_length_res.append(time() - start)
        
        results[length] = curr_length_res

def task2():
    random_prompt = "weazened acturience Euripidean glycosuria acuaesthesia gaybine amyxorrhea disheartening stanniferous Pyramidalism Hyblan thawless thiophenol siegeable glance followership sternpost private agraffee yus unpreach tributer parapodium unobjectionable sakeber squelching turbinated unicentral latibulize guardeen semiofficial asbestine"
    full_prompt = "My favorite holiday is Christmas, a time that transforms the ordinary into the magical. Cities and homes become wonderlands adorned with twinkling lights, evoking a sense of warmth and nostalgia. It's a period where the air carries the melody of carols, blending joy with the crispness of winter. The essence of Christmas goes beyond the material, fostering a spirit of generosity and togetherness. Families and friends gather, sharing meals and stories, creating memories that become treasured keepsakes. The excitement of giving and receiving gifts adds to the delight, but it's the moments of connection and shared laughter that truly encapsulate the holiday's charm. Amid the hustle, there's a profound peace in observing traditions and reflecting on the year. Christmas is not just a day but a feeling, embodying hope, love, and the magic of human kindness."
    full_prompt = full_prompt.replace("'", "")
    full_prompt = full_prompt.replace(",", "")
    
    def get_first_n_words(text, n):
        words = text.split()
        first_n_words = words[:n]
        return ' '.join(first_n_words)

    results = {}
    lengths = [32, 64, 128]
    t = 0.9

    prompts = {
        "prompt_random_32": random_prompt,
        "prompt_32": full_prompt,
        "prompt_64": full_prompt,
        "prompt_128": full_prompt,
    
    }

    
    for name, prompt in prompts.items():
        print(f"{name}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        max_tokens = name.split("_")[-1]
        if max_tokens in ["32", "64", "128"]:
            inputs['input_ids'] = inputs['input_ids'][:, :int(max_tokens)] 
            inputs['attention_mask'] = inputs['attention_mask'][:, :int(max_tokens)] 
        
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=1, do_sample=True, temperature=t)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # The modeling_llama now saves the attn matrices of all layers, here we move them to the right folder
        os.makedirs(name, exist_ok=True)
        filenames = [f for f in os.listdir(".") if f.endswith(".pkl")]
        for f in filenames:
            os.rename(f, os.path.join(name, f))
    

if __name__ == "__main__":
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Run the task of choice
    # task1_step2()
    # task1_step3()
    task2()
