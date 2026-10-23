import json


def parse_json_response(response: str, expected_type: type):
    """Extract and validate a JSON object or array from a model response."""
    if expected_type not in {dict, list}:
        raise TypeError("expected_type must be dict or list")

    text = response.split("```json")[-1].split("```")[0].strip()
    opening, closing = ("{", "}") if expected_type is dict else ("[", "]")
    if not text.startswith(opening):
        start = text.find(opening)
        end = text.rfind(closing)
        if start < 0 or end < start:
            raise ValueError("response did not contain the required JSON")
        text = text[start : end + 1]

    value = json.loads(text)
    if not isinstance(value, expected_type):
        raise ValueError(f"response must be a JSON {expected_type.__name__}")
    if expected_type is list and not all(isinstance(item, dict) for item in value):
        raise ValueError("response array must contain objects")
    return value
