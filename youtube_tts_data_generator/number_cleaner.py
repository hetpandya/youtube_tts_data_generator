import inflect
import re

inflect_engine = inflect.engine()
number_comma_regex = re.compile(r"([0-9][0-9\,]+[0-9])")
decimal_regex = re.compile(r"([0-9]+\.[0-9]+)")
currencies_regex = re.compile(r"[$₹£€¥]([0-9\.\,]*[0-9]+)")
ordinal_regex = re.compile(r"[0-9]+(st|nd|rd|th)")
number_regex = re.compile(r"[0-9]+")


def remove_commas(matches):
    return matches.group(1).replace(",", "")


def expand_decimal_point(matches):
    return matches.group(1).replace(".", " point ")


def misc_currency(currency_words, currency_words_single, parts, match):
    if len(parts) > 2:
        return match + " " + currency_words
    curr_amt = int(parts[0]) if parts[0] else 0
    amt = eval(match)
    curr_amt = match if amt > 0 and amt < 1 else curr_amt
    curr_unit = currency_words_single if curr_amt == 1 else currency_words
    return "%s %s" % (curr_amt, curr_unit)


def specific_currency(
    currency_words,
    currency_words_single,
    decimal_currency_words,
    decimal_currency_words_single,
    parts,
    match,
):
    if len(parts) > 2:
        return match + " " + currency_words  # Unexpected format
    curr_amt = int(parts[0]) if parts[0] else 0
    decimal_curr_amt = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if curr_amt and decimal_curr_amt:
        curr_unit = currency_words_single if curr_amt == 1 else currency_words
        decimal_curr_unit = (
            decimal_currency_words_single
            if decimal_curr_amt == 1
            else decimal_currency_words
        )
        return "%s %s, %s %s" % (
            curr_amt,
            curr_unit,
            decimal_curr_amt,
            decimal_curr_unit,
        )
    elif curr_amt:
        curr_unit = currency_words_single if curr_amt == 1 else currency_words
        return "%s %s" % (curr_amt, curr_unit)
    elif decimal_curr_amt:
        decimal_curr_unit = (
            decimal_currency_words_single
            if decimal_curr_amt == 1
            else decimal_currency_words
        )
        return "%s %s" % (decimal_curr_amt, decimal_curr_unit)
    else:
        return "zero " + currency_words


def expand_currency(matches):
    currency = matches.group(0)[0]
    match = matches.group(1)
    parts = match.split(".")
    if currency in ["$", "₹"]:
        if currency == "$":
            currency_words_single = "dollar"
            currency_words = "dollars"
            decimal_currency_words = "cent"
            decimal_currency_words_single = "cents"
        elif currency == "₹":
            currency_words_single = "rupee"
            currency_words = "rupees"
            decimal_currency_words = "paise"
            decimal_currency_words_single = "paisa"
        return specific_currency(
            currency_words,
            currency_words_single,
            decimal_currency_words,
            decimal_currency_words_single,
            parts,
            match,
        )
    else:
        if currency == "£":
            currency_words_single = "pound"
            currency_words = "pounds"
        elif currency == "€":
            currency_words_single = "euro"
            currency_words = "euros"
        elif currency == "¥":
            currency_words_single = "yen"
            currency_words = "yen"
        return misc_currency(currency_words, currency_words_single, parts, match)


def expand_ordinal(matches):
    return inflect_engine.number_to_words(matches.group(0))


def expand_number(matches):
    num = int(matches.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + inflect_engine.number_to_words(num % 100)
        elif num % 100 == 0:
            return inflect_engine.number_to_words(num // 100) + " hundred"
        else:
            return inflect_engine.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        return inflect_engine.number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(number_comma_regex, remove_commas, text)
    text = re.sub(currencies_regex, expand_currency, text)
    text = re.sub(decimal_regex, expand_decimal_point, text)
    text = re.sub(ordinal_regex, expand_ordinal, text)
    text = re.sub(number_regex, expand_number, text)
    return text
