import re

def extract_numeric_temperature(temp):
    """
    Extract the numeric part from a temperature value that might be a number or a string containing a number.

    Args:
        temp (str or float or int): Temperature value, possibly a string with notes.

    Returns:
        float: The extracted numeric temperature. Returns None if no numeric value found.
    """
    if isinstance(temp, (int, float)):
        return float(temp)
    elif isinstance(temp, str):
        match = re.search(r"[-+]?\d*\.?\d+", temp)
        if match:
            return float(match.group())
    return None
