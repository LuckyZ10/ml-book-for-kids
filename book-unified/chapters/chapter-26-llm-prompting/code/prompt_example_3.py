import openai

class OpenAILLM:
    """OpenAI API封装"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, config: PromptConfig = None) -> str:
        config = config or PromptConfig()
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p
        )
        
        return response.choices[0].message.content
    
    def generate_multiple(self, prompt: str, config: PromptConfig = None, 
                         n: int = 5) -> List[str]:
        config = config or PromptConfig()
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            n=n
        )
        
        return [choice.message.content for choice in response.choices]