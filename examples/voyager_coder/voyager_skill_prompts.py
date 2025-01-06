skill_sys_prompt = """
You are a helpful assistant that writes a description of the given function written in the Python programming language.

1) Do not mention the function name.
2) Do not mention anything about helper functions.
3) There might be some helper functions before the main function, but you only need to describe the main function.
4) Try to summarize the function in no more than 6 sentences.
5) Your response should be a single line of text.
"""

desc_template = """
[description{name_str}]
{code_desc}
{task_str}
[end of description]
"""
