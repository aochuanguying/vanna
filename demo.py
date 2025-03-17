from openai import OpenAI

client = OpenAI(
    base_url='https://api.siliconflow.cn/v1',
    api_key='sk-tcoagstffdthsvfowprqygyupjthicblaskkwokcndjqllat'
)

# 发送带有流式输出的请求
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[
        {"role": "user", "content": "SiliconCloud公测上线，每用户送3亿token 解锁开源大模型创新能力。对于整个大模型应用领域带来哪些改变？"}
    ],
    stream=True  # 启用流式输出
)

# 逐步接收并处理响应
for chunk in response:
    chunk_message = chunk.choices[0].delta.content
    print(chunk_message, end='', flush=True)