from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatOpenAI(
    base_url=os.environ.get("base_url"),
    api_key=os.environ.get("ARK_API_KEY"),
    model=os.environ.get("model"),
)


async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=llm,
    )
    await agent.run()

asyncio.run(main())
