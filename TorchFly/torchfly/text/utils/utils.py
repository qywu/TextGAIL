import os
import json
import regex as re
import logging

logger = logging.getLogger(__name__)

punctuations = [r" ?", r" !", r" .", r" ,", r" \"", r" \'", r" :", r" -", r" %"]


def fix_tokenized_punctuations(text):
    """
    I ' ve a cat . -> I've a cat.
    """
    text = re.sub(" (\u2018|\u2019|') ", r"\1", text)
    # fix puncutations
    for p in punctuations:
        text = text.replace(p, p[1:])

    text = text.replace("\' ", "\'")
    text = text.replace("\" ", "\"")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace(",\'", ",\' ")
    text = text.replace(",\"", ",\" ")

    return text