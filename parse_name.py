from name_parsing_chain import get_name_parsing_chain
from name_parsing_template import get_system_template, get_examples, get_human_template

def parse_name(user_list: list[str]) -> list[str]:
    return get_name_parsing_chain(
        get_system_template(),
        get_examples(),
        get_human_template()
    ).invoke({"user_list": user_list})