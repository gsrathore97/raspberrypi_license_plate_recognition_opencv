# -*- coding: utf-8 -*-
import pytesseract
def ocr(plate):
    text = pytesseract.image_to_string(plate,config="",lang="eng")
    return text

def check_if_string_in_file(file_name, string_to_search):
    with open(file_name, 'r') as read_obj:
        for line in read_obj:
            if string_to_search in line:
                return True
    return False

