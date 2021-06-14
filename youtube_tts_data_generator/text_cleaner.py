import re
from unidecode import unidecode
from .number_cleaner import normalize_numbers


class Cleaner(object):
    def __init__(self):
        self._whitespace_re = re.compile(r"\s+")
        self._abbreviations = [
            (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
            for x in [
                ("mrs", "misess"),
                ("mr", "mister"),
                ("dr", "doctor"),
                ("st", "saint"),
                ("co", "company"),
                ("jr", "junior"),
                ("maj", "major"),
                ("gen", "general"),
                ("drs", "doctors"),
                ("rev", "reverend"),
                ("lt", "lieutenant"),
                ("hon", "honorable"),
                ("sgt", "sergeant"),
                ("capt", "captain"),
                ("esq", "esquire"),
                ("ltd", "limited"),
                ("col", "colonel"),
                ("ft", "fort"),
                ("rs", "rupees"),
            ]
        ]

    def expand_abbreviations(self, text):
        for regex, replacement in self._abbreviations:
            text = re.sub(regex, replacement, text)
        return text

    def collapse_whitespace(self, text):
        return re.sub(self._whitespace_re, " ", text)

    def clean_english_text(self, text):
        """Pipeline for English text, including number and abbreviation expansion."""
        text = normalize_numbers(text)
        text = unidecode(text)
        text = text.lower()
        text = self.expand_abbreviations(text)
        text = self.collapse_whitespace(text)
        return text
