import fitz

my_path = "report.pdf"

doc = fitz.open(my_path)

for page in doc:
    text = page.get_text()
    print(text)

output = page.get_text("blocks")

for page in doc:
    output = page.get_text("blocks")
    previous_block_id = 0 # Set a variable to mark the block id
    for block in output:
        if block[6] == 0: # We only take the text
            if previous_block_id != block[5]:
                # Compare the block number
                print("\n")
            print(block[4])

from unidecode import unidecode
output = []
for page in doc:
    output += page.get_text("blocks")
previous_block_id = 0 # Set a variable to mark the block id
for block in output:
    if block[6] == 0: # We only take the text
        if previous_block_id != block[5]: # Compare the block number
            print("\n")
        plain_text = unidecode(block[4])
        print(plain_text)

block_dict = {}
page_num = 1
for page in doc: # Iterate all pages in the document
    file_dict = page.get_text('dict') # Get the page dictionary
    block = file_dict['blocks'] # Get the block information
    block_dict[page_num] = block # Store in block dictionary
    page_num += 1 # Increase the page value by 1


import pandas as pd
import re
spans = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'text', 'tag'])
rows = []
for page_num, blocks in block_dict.items():
    for block in blocks:
        if block['type'] == 0:
            for line in block['lines']:
                for span in line['spans']:
                    xmin, ymin, xmax, ymax = list(span['bbox'])
                    font_size = span['size']
                    text = unidecode(span['text'])
                    span_font = span['font']
                    is_upper = False
                    is_bold = False
                    if "bold" in span_font.lower():
                        is_bold = True
                    if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                        is_upper = True
                    if text.replace(" ","") !=  "":
                        rows.append((xmin, ymin, xmax, ymax, text,                              is_upper, is_bold, span_font, font_size))
                        span_df = pd.DataFrame(rows, columns=['xmin','ymin','xmax','ymax', 'text', 'is_upper','is_bold','span_font', 'font_size'])

