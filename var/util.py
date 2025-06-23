from pydantic import BaseModel, SkipValidation
from pydantic_settings import BaseSettings

from functools import cache
from transformers import AutoTokenizer, AutoModelForCausalLM


# Refrain from calling these inside the notebook if cache is not available. 
# Takes too long. Use the console instead
from contextlib import redirect_stdout
import io


class Settings(BaseSettings):
    hf_finegrained_token: str
    hf_readonly_token: str

    # The decoration order counts. Eventually it must be a classmethod
    @classmethod
    @cache
    def make(env_path: str=".env") -> Settings:
        return Settings()


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
                    # NEVER forget this for causal models
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