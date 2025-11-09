import re

def extract_numbers(string):
  numbers = re.findall(r'\d+', string)
  return [int(num) for num in numbers]

def to_string_with_2_digits(number):
  return "{:02d}".format(number)