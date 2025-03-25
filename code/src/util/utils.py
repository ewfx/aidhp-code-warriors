import json

def parse_json(input_str):
    try:
        return json.loads(input_str), None
    except json.JSONDecodeError as e:
        return None, str(e)
