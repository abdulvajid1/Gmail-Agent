from typing import Any

from langchain_core.language_models import BaseChatModel
from utils import setup_openai_client
from langchain_core.messages import convert_to_openai_messages
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk

class CustomLLM(BaseChatModel):
    client: Any = None
    def __init__(self, openai_url:str = "http://localhost:11434/v1"):
        super().__init__()
        self.client = setup_openai_client(openai_url)
    
    @property
    def _llm_type(self) -> str:
        return "llama3.1"
    
    def _generate(self, messages, stop=None, **kwargs):
        messages = convert_to_openai_messages(messages)

        response = self.client.chat.completions.create(
            messages=messages,
            model=self._llm_type,
            max_tokens=kwargs.get('max_tokens', 100)
        )
        chat_generated = ChatGeneration(message=AIMessage(response.choices[0].message.content))
        return ChatResult(generations=[chat_generated])
        
    def _stream(self, messages, stop = None, run_manager = None, **kwargs):
        messages = convert_to_openai_messages(messages)
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self._llm_type,
            max_tokens=kwargs.get('max_tokens', 100),
            stream=True
        )
        
        for i in response:
            yield ChatGenerationChunk(message=AIMessageChunk(content=i.choices[0].delta.content))
        
        
    

if __name__ == "__main__":
    llm = CustomLLM()
    msg = [
        SystemMessage('you are a good man'),
        HumanMessage("who are you")
    ]
    # print(llm.invoke(msg).content)
    
    for i in llm.stream(msg):
        print(i.content, end='', flush=True)