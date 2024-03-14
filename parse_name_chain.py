from operator import itemgetter

from langchain_core.runnables import RunnableWithFallbacks, RunnableSerializable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

def create_chat_model_with_fallbacks() -> RunnableWithFallbacks:
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0,
        convert_system_message_to_human=True
    ).with_fallbacks(
        [
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0
            )
        ]
    )

def create_chat_prompt_template_with_few_shot(system_template: str, examples: list[dict[str, str]]) -> FewShotChatMessagePromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            FewShotChatMessagePromptTemplate(
                example_prompt=ChatPromptTemplate.from_messages(
                    [
                        ("human", "{input}"),
                        ("ai", "{output}")
                    ]
                ),
                examples=examples
            ),
            ("human", "{input}\n\nAI: ")
        ]
    )

def get_parse_name_chain(system_template: str, examples: list[dict[str, str]]) -> RunnableSerializable:
    return (
        {
            "input": itemgetter("input")
        }
        | create_chat_prompt_template_with_few_shot(system_template, examples)
        | create_chat_model_with_fallbacks()
        | CommaSeparatedListOutputParser()
    )