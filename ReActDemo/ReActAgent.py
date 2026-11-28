import re

from tools.ToolsManager import ToolsManager
from tools.Search import search
from LLMClient import LLMClient


# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: LLMClient, toolsManager: ToolsManager,  max_steps: int = 5):
        self.llm_client = llm_client
        self.tools_manager = toolsManager
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """

        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tools_manager.getAvailableTools()
            history_str = "\n".join(self.history)

            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print(f"警告:无效的Action格式: {action}")
                continue

            if tool_name == "Finish":
                print(f"🎉 最终答案: {tool_input}")
                return tool_input

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tools_manager.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具

            print(f"👀 观察: {observation}")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """
        解析LLM的输出，提取Thought和Action。
        """
        # 只读取第一个字段块，遇到下一个协议字段即停止。
        # 使用 \Z 表示整个文本末尾，避免 MULTILINE 下的 $ 提前结束多行内容。
        flags = re.DOTALL | re.MULTILINE
        boundary = r"(?=^[ \t]*(?:Thought|Action|Observation):|\Z)"
        thought_match = re.search(r"^[ \t]*Thought:[ \t]*(.*?)" + boundary, text, flags)
        action_match = re.search(r"^[ \t]*Action:[ \t]*(.*?)" + boundary, text, flags)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        # 完整匹配已截取的 Action，保留多行内容和参数内部的方括号。
        match = re.fullmatch(r"(\w+)\[(.*)\]", action_text.strip(), re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None

