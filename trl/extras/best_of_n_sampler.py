# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Union

import torch
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast, set_seed

from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper

class BestOfNSampler:
    def __init__(
        self,
        queries_to_scores: Callable[[list[str]], list[float]],
        n_candidates: int = 1,
    ) -> None:
        r"""
        Initialize the sampler for best-of-n generation

        Args:
            queries_to_scores (`Callable[[list[str]], list[float]]`):
                Callable that takes a list of generated texts and returns the associated reward scores
            n_candidates (`int`):
                Number of candidates to return for each query
        """
        self.queries_to_scores = queries_to_scores
        self.n_candidates = n_candidates

    def select_best(
        self,
        generated_texts: list[list[str]],
        queries_to_scores_args: Any
    ) -> list[list[str]]:
        r"""
        Select the best candidates from generated texts based on scores

        Args:
            generated_texts (`list[list[str]]`):
                List of lists of generated texts (outer list for queries, inner list for samples)

        Returns:
            list[list[str]]: A list of lists of selected texts
        """
        result = []
        for samples in generated_texts:
            scores = torch.tensor(self.queries_to_scores(samples, **queries_to_scores_args))
            selected = [samples[i] for i in scores.topk(self.n_candidates).indices]
            result.append(selected)
        return result

    @staticmethod
    def generate_samples(
        model: PreTrainedModelWrapper,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        tokenized_query: Union[list[int], torch.Tensor, list[torch.Tensor], list[list[int]]],
        max_new_tokens_sampler: Any,
        sample_size: int,
        generation_config: Optional[GenerationConfig] = None,
        skip_special_tokens: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **generation_kwargs,
    ) -> list[list[str]]:
        r"""
        Generate samples for input queries

        Args:
            model (`PreTrainedModelWrapper`):
                The pretrained model to use for generation
            tokenizer (`PreTrainedTokenizer` or `PreTrainedTokenizerFast`):
                Tokenizer associated with the pretrained model
            tokenized_query (`list[int]` or `torch.Tensor` or `list[torch.Tensor]` or `list[int]`):
                represents either a single tokenized query or a batch of tokenized queries
            max_new_tokens_sampler (`Any`):
                Sampler used to sample the max_new_tokens for each generated text
            sample_size (`int`):
                Number of samples to generate for each query
            generation_config (`GenerationConfig`, *optional*):
                Generation config passed to the underlying model's `generate` method
            skip_special_tokens (`bool`):
                Whether to remove the special tokens from the output
            device (`str` or `torch.device`, *optional*):
                The device on which the model will be loaded
            **generation_kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's `generate` method

        Returns:
            list[list[str]]: A list of lists of generated texts
        """
        if not isinstance(model, (SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )

        queries = None

        if isinstance(tokenized_query, torch.Tensor) and tokenized_query.ndim == 1:
            queries = tokenized_query.unsqueeze(0)
        elif isinstance(tokenized_query, list):
            element_type = type(tokenized_query[0])
            if element_type is int:
                queries = torch.tensor(tokenized_query).unsqueeze(0)
            elif element_type is torch.Tensor:
                queries = [tensor.reshape((1, -1)) for tensor in tokenized_query]
            else:
                queries = [torch.tensor(query).reshape((1, -1)) for query in tokenized_query]

        result = []
        for query in queries:
            queries = query.repeat((sample_size, 1))
            output = model.generate(
                queries.to(device),
                max_new_tokens=max_new_tokens_sampler(),
                generation_config=generation_config,
                **generation_kwargs,
            ).squeeze()
            output = tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)
            result.append(output)

        return result
