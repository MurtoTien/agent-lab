from ReflectionAgent import ReflectionAgent
from LLMClient import LLMClient

if __name__ == '__main__':
    try:
        llm_client = LLMClient()

        task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
        agent = ReflectionAgent(llm_client=llm_client)
        agent.run(task=task)
        
    except ValueError as e:
        print(e)
