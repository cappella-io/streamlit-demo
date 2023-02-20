import openai 

class Chatbot :
    
    def __init__(
        self, 
        api_key : str,
        temperature : float = 1,
        max_tokens : int = 2000,
        presence_penalty : float = 0,
        frequency_penalty : float = 0,
        engine : str = "text-davinci-003"
    ) :
        openai.api_key = api_key
        
        self.temperature = temperature 
        self.max_tokens = max_tokens 
        self.presence_penalty = presence_penalty 
        self.frequency_penalty =frequency_penalty
        self.engine = engine
    
    def query(
        self,
        prompt : str
    ) :
        completion = openai.Completion.create(
            engine = self.engine,
            prompt = prompt,
            temperature = self.temperature,
            max_tokens = self.max_tokens, 
            presence_penalty = self.presence_penalty,
            frequency_penalty = self.frequency_penalty
        )
        
        if completion is None or completion['choices'] is None or len(completion['choices']) == 0 :
            return "No answer is generated !"
        
        answer = completion['choices'][0]['text']
        
        return answer 
        
        
        
        
        
        