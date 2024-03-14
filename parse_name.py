from parse_name_chain import get_parse_name_chain
from parse_name_template import make_system_template, make_examples

def parse_name(user_list: list[str]) -> list[str]:
    return get_parse_name_chain(
        make_system_template(),
        make_examples()
    ).invoke({"input": user_list})