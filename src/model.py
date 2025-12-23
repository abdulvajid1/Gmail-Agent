from typing import Any

from langchain_core.language_models import BaseChatModel
from utils import setup_openai_client
from langchain_core.messages import convert_to_openai_messages
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import tool

from prompts import TOOL_PROMPT

class CustomLLM(BaseChatModel):
    client: Any = None
    tool_prompt : Any = None
    tools: Any = None
    def __init__(self, openai_url:str = "http://localhost:11434/v1", tool_prompt=None, tools=None):
        super().__init__()
        self.client = setup_openai_client(openai_url)
        self.tool_prompt = tool_prompt
        self.tools = tools
        
    @property
    def _llm_type(self) -> str:
        return "llama3.1"
    
    def _generate(self, messages, stop=None, **kwargs):
        
        if self.tools:    
            last_user_message = messages.pop().content
            new_user_msg = self.tool_prompt + last_user_message
            messages.append(HumanMessage(content=new_user_msg))
            
        messages_dict = convert_to_openai_messages(messages)
        
        response = self.client.chat.completions.create(
            messages=messages_dict,
            model=self._llm_type,
            max_tokens=kwargs.get('max_tokens', 100)
        )
        
        chat_generated = ChatGeneration(message=AIMessage(content=response.choices[0].message.content))
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
            
    def bind_tools(self, tools, *, tool_choice = None, **kwargs):
        
        openai_tool_json = [convert_to_openai_tool(tool) for tool in tools]
        tool_prompt = TOOL_PROMPT.format(tools=openai_tool_json)
        print(tool_prompt)
        
        return CustomLLM(tool_prompt=tool_prompt, tools=tools)
    
    def with_structured_output(self, schema, *, include_raw = False, **kwargs):
        pass
        

# TODO : with structured output , bind_tool()

if __name__ == "__main__":
    llm = CustomLLM()
    msg = [
        SystemMessage('you are a good man'),
        HumanMessage("who are you")
    ]
    # print(llm.invoke(msg).content)
    
    for i in llm.stream(msg):
        print(i.content, end='', flush=True)
        
    @tool    
    def add(a: int, b: int):
        """Add two numbers"""
        return a+b
    
    llm_with_tool = llm.bind_tools([add])
    print(llm_with_tool.invoke("add 3 and 2"))