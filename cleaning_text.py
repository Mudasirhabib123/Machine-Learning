import re
def capatelizer(s: str) -> str:
    return s.upper()

def replace_x(s: str) -> str:
    return re.sub(r'[a-zA-Z]', 'X',s)


text_data = [
    "       Interrobang. By Aishwarya Henriette"
    "Parking And Going. By Karl Gautier         ",
    "       Today Is The night. By Jarek Prakash       "
]

strip_spaces = [string.strip() for string in text_data]

# removing periods

periods_less = [string.replace('.','') for string in strip_spaces]
periods_less = [capatelizer(string) for string in periods_less]
periods_less = [replace_x(string) for string in periods_less]

print(periods_less)
