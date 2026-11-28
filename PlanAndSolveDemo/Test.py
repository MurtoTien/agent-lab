from PlanAndSolveAgent import PlanAndSolveAgent
from LLMClient import LLMClient

if __name__ == '__main__':
    try:
        llm_client = LLMClient()

        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        agent = PlanAndSolveAgent(llm_client=llm_client)
        agent.run(question=question)
        
    except ValueError as e:
        print(e)