# Enable the use of classname in classmethods
from __future__ import annotations
from contextlib import redirect_stdout
from functools import cache
import io
import os
import re
from typing import Optional, List

from pydantic import BaseModel, SkipValidation, Field
from pydantic_settings import BaseSettings
from transformers import AutoTokenizer, AutoModelForCausalLM



class Settings(BaseSettings, extra="ignore"):
    hf_readonly_token:Optional[str]=Field(alias="HF_READONLY_TOKEN", default=None)
    hf_finegrained_token: Optional[str]=Field(alias="HF_FINEGRAINDED_TOKEN", default=None)
    openai_api_key: str=Field(alias="OPENAI_API_KEY", default="christos")

    # The decoration order counts. Eventually it must be a classmethod
    @classmethod
    @cache
    def make(env_path: str=".env") -> Settings:        
        return Settings(_env_file=(env_path if not os.isfile(env_path) else None))


class ModelConfig(BaseModel):
    # This is required to set the member typehints to Auto*
    model_config = dict(arbitrary_types_allowed=True)

    model_id: str
    tokenizer: SkipValidation[AutoTokenizer]
    model: SkipValidation[AutoModelForCausalLM]
    
    # The order of decorators does matter
    @classmethod
    @cache
    def from_pretrained(cls, model_id: str):
        # Dump ModelConfig ctor logging to /dev/null
        with redirect_stdout(io.StringIO()):
            return ModelConfig(            
                model_id=model_id,
                tokenizer=AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=model_id, 
                    # NEVER forget this for causal models (only if batched, otherwise makes no dofference)
                    padding_side="left"
                ),
                model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id)
            )
    
    def generate(self, query: str, max_new_tokens=512, temperature=.7, top_p=.9) -> str:
        inputs = self.tokenizer(query, return_tensors="pt")
        token_ids = self.model.generate(
            **inputs, 
            # Check https://huggingface.co/docs/transformers/llm_tutorial
            do_sample=True, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_p=top_p,
            num_beams=4
        )
        return self.tokenizer.batch_decode(
            token_ids, 
            skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    

# The default pattern used below splits sentences.
# We do not use capturing parentheses so that split characters are excluded from the result
# https://docs.python.org/3.12/library/re.html#re.split

# To test if a split pattern works, compare with the original strin using capturing parentheses, e.g.
# SYSTEM_PROMPT == ''.join(re.split(f"({splitpat})", SYSTEM_PROMPT))

def pattern_split(
    text: str, 
    patternlist: List[re.Pattern]=[re.compile(r"(?<=[.:;!?])\s+")],
    joinstr: str=os.linesep
) -> str:
    ''' Split a (long) string into multiple lines 
        (or do sth more general using multiple split patterns and 
        join string).
        I use this below to split LLM outputs into short lines.
    '''
    inputs, outputs = [text], []
    for pat in patternlist:
        _ = [outputs.extend(pat.split(s)) for s in inputs]
        inputs, outputs = outputs, []
    return joinstr.join(inputs)

# Testing
# print(pattern_split(text="One. Two; and three"))