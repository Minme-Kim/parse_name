from parse_name_chain import get_parse_name_chain
from parse_name_template import get_system_template, get_examples, get_human_template

def parse_name(user_list: list[str]) -> list[str]:
    return get_parse_name_chain(
        get_system_template(),
        get_examples(),
        get_human_template
    ).invoke({"user_list": user_list})