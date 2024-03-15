from operator import itemgetter

from langchain_core.runnables import RunnableWithFallbacks, RunnableSerializable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
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

def create_chat_prompt_template_with_few_shot(system_template: str, examples: list[dict[str, str]], human_template: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            FewShotChatMessagePromptTemplate(
                example_prompt=ChatPromptTemplate.from_messages(
                    [
                        ("human", "사용자 리스트: {user_list}"),
                        ("ai", "수정된 리스트: {modified_list}")
                    ]
                ),
                examples=examples
            ),
            ("human", human_template)
        ]
    )

def get_parse_name_chain(system_template: str, examples: list[dict[str, str]], human_template: str) -> RunnableSerializable:
    return (
        {
            "user_list": itemgetter("user_list")
        }
        | create_chat_prompt_template_with_few_shot(system_template, examples, human_template)
        | create_chat_model_with_fallbacks()
        | CommaSeparatedListOutputParser()
    )