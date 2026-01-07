import os
import time
from typing import Optional, List

from dotenv import load_dotenv
from numpy.matlib import empty
from openai import OpenAI


load_dotenv()

class LLM:
    def __init__(self,
                 model,
                 api_key:Optional[str]=None,
                 base_url:Optional[str]=None,
                 provider:Optional[str]=None,
                 **kwargs):
        if provider.lower() == "chatanywhere":
            try:
                print(f"正在调用免费的{model}大模型通过chatanywhere")
                self.model = model
                self.api_key = os.getenv("CHATANYWHERE_API")
                self.base_url = os.getenv("CHATANYWHERE_BASE_URL")
                self.temperature = kwargs.get("temperature", 0.7)
                self.max_token = kwargs.get("max_token")
                self.timeout = kwargs.get("timeout", 60)
                self.client = OpenAI(api_key=self.api_key,base_url=self.base_url,timeout=60)
            except Exception as e:
                return f"调用模型出错{e}"
        #根据模型名称选择具体支持模型
        else:
            print(f"您正在使用自己的apikey调用{model}模型，请注意token消耗！")
            if model:
                if "gemini" in model.lower():
                    self.model = os.getenv("GEMINI_LLM")
                    self.api_key = os.getenv("GEMINI_API_KEY")
                    if self.model is None or self.api_key is None:
                        raise ValueError(f"请检查你的{model} model_name或者{model} API_KEY是否在环境中正确设置")
                    print(f"您正在使用 🚀 {self.model} 模型")
                elif "claude" in model.lower():
                    self.model = os.getenv("CLAUDE_LLM")
                    self.api_key = os.getenv("CLAUDE_API_KEY")
                    if self.model is None or self.api_key is None:
                        raise ValueError(f"请检查你的{model} model_name或者{model} API_KEY是否在环境中正确设置")
                    print(f"您正在使用 🚀 {self.model} 模型")
                elif "qwen" in model.lower():
                    self.model = os.getenv("QWEN_LLM")
                    self.api_key = os.getenv("QWEN_API_KEY")
                    if self.model is None or self.api_key is None:
                        raise ValueError(f"请检查你的{model} model_name或者{model} API_KEY是否在环境中正确设置")
                    print(f"您正在使用 🚀 {self.model} 模型")
                else:
                    raise ValueError("您所输入的模型暂不支持")
            else:
                raise ValueError("请您正确配置你的model_name,可选模型:[Gemini,Claude,Qwen]")


            # self.model = model or os.getenv("LLM_MODEL_NAME") or "gemini-2.5-flash"
            self.temperature = kwargs.get("temperature",0.7)
            self.max_token  =kwargs.get("max_token")
            self.timeout = kwargs.get("timeout",60)
            #OpenAI的baseurl
            self.base_url = base_url or os.getenv("BASE_URL")
            #使用获取的参数创建OpenAI客户端实例
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,timeout=self.timeout)

    def invoke(self,messages:list[dict[str,str]],**kwargs):
        """
        非流失调用LLM，直接返回完整响应
        :param messages:
        :param kwargs:
        :return:
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_token,
                **{k:v for k,v in kwargs.items() if k not in ['temperature','max_token']}
            )
            # print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            raise e

    def stream_invoke(self,messages:list[dict[str,str]],**kwargs):
        """
        流失调用LLM,实现打字机效果
        :param messages:
        :param kwargs:
        :return:
        """
        try:
            """两种不同实现方式"""
            with self.client.chat.completions.stream(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_token,
            ) as stream:
                for event in stream:
                    if event.type == "content.delta":
                        print(event.delta, end="", flush=True)


            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=messages,
            #     temperature=self.temperature,
            #     max_tokens=self.max_token,
            #     stream=True,
            #     **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_token']}
            # )
            # for chunk in response:
            #     if hasattr(chunk, "choices") and chunk.choices:
            #         delta = chunk.choices[0].delta
            #         if delta:
            #             for dict_content in delta:
            #                 content = dict_content[1]
            #                 if content is None:
            #                     continue
            #                 for char in content:
            #                     print(char, end="", flush=True)
                                # time.sleep(0.5)
                # else:
                #     break
        except Exception as e:
            raise e




#测试
if __name__ == "__main__":
    my_llm = LLM(model="gpt-5-mini",provider="chatanywhere")
    message=[{"role":"user","content":"请你介绍一下自己"}]
    my_llm.stream_invoke(messages=message)


