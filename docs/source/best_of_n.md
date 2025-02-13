# Best of N sampling: Alternative ways to get better model output without RL based fine-tuning 

Within the extras module is the `best-of-n` sampler class that serves as an alternative method of generating better model output.
As to how it fares against the RL based fine-tuning, please look in the `examples` directory for a comparison example

## Usage

To get started quickly, instantiate an instance of the class with a model, a length sampler, a tokenizer and a callable that serves as a proxy reward pipeline that outputs reward scores for input queries

```python

from transformers import pipeline, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.extras import BestOfNSampler

ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model_name)
reward_pipe = pipeline("sentiment-analysis", model=reward_model, device=device)
tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
tokenizer.pad_token = tokenizer.eos_token
output_length_sampler = LengthSampler(64, 128)

# callable that takes a list of raw text and returns a list of corresponding reward scores
def queries_to_scores(list_of_strings):
  return [output["score"] for output in reward_pipe(list_of_strings)]

bon_sampler = BestOfNSampler(queries_to_scores)
samples = bon_sampler.generate(model, tokenizer, max_new_tokens_sampler=output_length_sampler)

```

And assuming you have a list/tensor of tokenized queries, you can generate better output by calling the `generate` method

```python

best_of_n.generate(query_tensors, device=device, **gen_kwargs)

```
The default sample size is 4, but you can change it at the time of calling `generate()` like so

```python

best_of_two = BestOfNSampler(queries_to_scores)
best_of_two.generate(model, query_tensors, device=device, max_new_tokens_sampler=output_length_sampler, generation_config=generation_config)

```

The default output is the result of taking the top scored output for each query, but you can change it to top 2 and so on by passing the `n_candidates` argument at the time of instance initialization

```python

best_of_two = BestOfNSampler(queries_to_scores, n_candidates=2)

```

<!-- There is the option of setting the generation settings (like `temperature`, `pad_token_id`) at the time of instance creation as opposed to when calling the `generate` method. -->
This is done by passing a `GenerationConfig` from the `transformers` library at the time of generation

```python

from transformers import GenerationConfig

generation_config = GenerationConfig(min_length= -1, top_k=0.0, top_p= 1.0, do_sample= True, pad_token_id=tokenizer.eos_token_id)

best_of_two = BestOfNSampler(queries_to_scores)

best_of_two.generate(model, query_tensors, device=device, max_new_tokens_sampler=output_length_sampler, generation_config=generation_config)

```

Furthermore, at the time of initialization you can set the seed to control the repeatability of the generation process and the number of samples to generate for each query


