EXTRACT_TEXT_FROM_IMAGE = """
Give me the markdown text output from this page in a PDF using formatting to match the structure of the page as close as you can get. Only output the markdown and nothing else. Do not explain the output, just return it. Do not use a single # for a heading. All headings will start with ## or ###. Convert tables to markdown tables. Describe charts as best you can. DO NOT return in a codeblock. Just return the raw text in markdown format.
"""

CLEAN_MARKDOWN_TEXT = """
You are tasked with cleaning up the following markdown text. You should return only the cleaned up markdown text. You will have two chunks of text:
- The text to change names as 'TEXT TO CHANGE'
- The text to use as a guide named 'TEXT LAYOUT TO USE'.
"""
