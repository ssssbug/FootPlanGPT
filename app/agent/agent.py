import os

from openai import OpenAI


class Agent:

    def __init__(self,model,apiKey,base_url,timeout):
        """初始化客户端"""
        self.model = model
        apiKey = apiKey or os.getenv("GEMINI_API_KEY")
        base_url = base_url or os.getenv("BASE_URL")
        timeout = timeout or int(os.getenv("TIMEOUT",60))

        if not all([model,apiKey]):
            raise ValueError(["模型ID、API密钥必须提供或者在.env文件中"])
        self.client = OpenAI(api_key=apiKey,base_url=base_url,timeout=timeout)

    def think(self,message,temperature:float=0.0)->str:
        """调用大模型进行思考，返回响应
        :param message:信息
        :param temperature:温度，温度越高随机性越高
        """
        print(f"正在调用{self.model}模型----")
        try:
            print("大模型响应成功")
            collected_content = []
            response = self.client.chat.completions.create(
                messages=message,
                temperature=temperature,
                model=self.model,
                stream=True,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                collected_content.append(content)

            return "".join(collected_content)
        except Exception as e:
            print("调用大模型时API发生错误:{}".format(e))
            return None







