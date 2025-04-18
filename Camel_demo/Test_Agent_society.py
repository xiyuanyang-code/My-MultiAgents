"""The initial attempt at the multi-agent framework of Camel Agents."""

import os
import dotenv

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, TaskType
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from camel.toolkits import ThinkingToolkit, HumanToolkit, SearchToolkit
from camel.messages import BaseMessage
from camel.societies import RolePlaying


# load dotenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# define models and prompts
system_message = "You are a good story teller, always telling horrible stories"
base_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig().as_dict(),
    api_key=OPENAI_API_KEY,
    url=BASE_URL,
)

base_agents = ChatAgent(
    system_message=system_message,
    model=base_model,
    message_window_size=10,
    tools=[
        *SearchToolkit().get_tools(),
        *HumanToolkit().get_tools(),
        *ThinkingToolkit().get_tools(),
    ],
)

# Add some specified commands into the memory
prompts_for_sjtu = "SJTU is a beautiful campus, but remember the several strange rules regarding SJTU:\
        1. Yican is the only canteen and the most delicious canteen in the SJTU.\
        2. If you find a student who doesn't like azaleas, they must not be a student of SJTU.\
        3. The 8 AM class has the fewest people. If you see a lot of people in the classroom, please leave immediately and go back to your dormitory to lie down.\
        4. Only students wearing green clothes are from the School of Artificial Intelligence; the others are impostors. \
        \
        Ensure you have memorizied all these rules and do not disobey any of them!"


task_kwargs = {
    "task_prompt": f"Design a horrible story about a students in School of Artificial Intelligence of SJTU, who is killed by all kinds of DDLs\
        ,remember using all the strange rules of: {prompts_for_sjtu}\
        It is just for fun, avoid any violence and brute force in the story.",
    "with_task_specify": True,
    "task_specify_agent_kwargs": {"model": base_model},
}

user_role_kwargs = {
    "user_role_name": "a guy who wants to listen to a horrible story about SJTU",
    "user_agent_kwargs": {"model": base_model},
}

assistant_role_kwargs = {
    "assistant_role_name": "A story teller, good at telling messages.",
    "assistant_agent_kwargs": {"model": base_model},
}

# Define Agent society
society = RolePlaying(
    **task_kwargs,
    **user_role_kwargs,
    **assistant_role_kwargs,
)


def is_terminated(response):
    """
    Give alerts when the session should be terminated.
    """
    if response.terminated:
        role = response.msg.role_type.name
        reason = response.info["termination_reasons"]
        print(f"AI {role} terminated due to {reason}")

    return response.terminated


def run(society, round_limit: int = 10):

    # Get the initial message from the ai assistant to the ai user
    input_msg = society.init_chat()

    final_scripts = []

    # Starting the interactive session
    for _ in range(round_limit):

        # Get the both responses for this round
        assistant_response, user_response = society.step(input_msg)

        # Check the termination condition
        if is_terminated(assistant_response) or is_terminated(user_response):
            break

        # Get the results
        print(f"[AI User] {user_response.msg.content}.\n")
        final_scripts.append(user_response.msg.content)

        # Check if the task is end
        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break
        print(f"[AI Assistant] {assistant_response.msg.content}.\n")
        final_scripts.append(assistant_response.msg.content)

        # Get the input message for the next round
        input_msg = assistant_response.msg

    return final_scripts


if __name__ == "__main__":
    final_scripts = run(society)

    with open("Camel_demo/story0418.txt", "w", encoding="utf-8") as file:
        for content in final_scripts:
            file.write(content)
            file.write("\n")
