def get_system_template() -> str:
    return """###임무###

- 당신의 임무는 사용자 리스트에서 '이름'만을 추출해내는 것이다.
(이 임무를 잘 수행했을 시에는, 100달러의 팁이 제공될 것이다. 하지만 그렇지 못했을 시에는, 법적 불이익이 있을 것이다.)

###조건###

- '이름'만을 추출해야만 한다.
- '공백'을 제거해야만 한다.
- 학번이 적혀있을 경우, '학번'을 출력해야만 한다.
- 하나하나 '단계별'로 생각해야만 한다.

###예시###"""

def get_examples() -> list[dict[str, str]]:
    return [
        {
            "input": "신윤철B반, 김승아, 202345038, 박민준",
            "output": "신윤철, 김승아, 202345038, 박민준"
        },
        {
            "input": "컴시과 1-A 조준, 3-b 전소영, 김 민 호, 김태 민",
            "output": "조준, 전소영, 김민호, 김태민"
        },
        {
            "input": "허원, 김경서, 장예준MT, 임형주",
            "output": "허원, 김경서, 장예준, 임형주"
        },
        {
            "input": "박재연, 2-A김서정, 2-A 김재현, 202245001",
            "output": "박재연, 김서정, 김재현, 202245001"
        },
        {
            "input": "이예찬, B반김태윤, 김강영, 2학년A반김경민",
            "output": "이예찬, 김태윤, 김강영, 김경민"
        },
        # {
        #     "input": "",
        #     "output": ""
        # },
        # {
        #     "input": "",
        #     "output": ""
        # },
        # {
        #     "input": "",
        #     "output": ""
        # },
        # {
        #     "input": "",
        #     "output": ""
        # },
        # {
        #     "input": "",
        #     "output": ""
        # }
    ]

def get_human_template() -> str:
    return """{user_list}

AI: """