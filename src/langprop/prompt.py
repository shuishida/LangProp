import re
import sys
from io import StringIO


class CapturePrint(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def exec_py_code(_py_code, _globals_dict):
    for _k, _v in _globals_dict.items():
        globals()[_k] = _v
    try:
        with CapturePrint() as _o:
            exec(_py_code, globals(), globals())
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        print("There was an error during execution of python code parsed from text. The python code was as follows:")
        print(_py_code)
        raise e
    for _k, _v in globals().items():
        _globals_dict[_k] = _v
    return "\n".join(_o)


def replace_variables(line, locals_dict):
    result = line
    # detect double braces {{ }}
    for var in re.findall(r'\{\{[^{}]*\}\}', line):
        result = result.replace(var, exec_py_code("print(" + str(var[2:-2]) + ")", locals_dict))
    return result


def parse_template(template: str, **locals_dict):
    results = ""
    py_code = ""
    inside_py_block = False
    for line in template.split("\n"):
        line = line.strip("\n")
        if line[:2] == "$ ":
            py_code += line[2:] + "\n"
        elif line[:6] == "$begin":
            inside_py_block = True
        elif line[:4] == "$end":
            inside_py_block = False
        elif line[:1] == "#":
            continue
        else:
            if inside_py_block:
                py_code += replace_variables(line, locals_dict) + "\n"
            else:
                if py_code:
                    results += exec_py_code(py_code, locals_dict)
                    py_code = ""
                results += replace_variables(line, locals_dict) + "\n"
    if py_code:
        results += exec_py_code(py_code, locals_dict)
    return results


def extract_block(string: str, start="```python", end="```", return_list=False):
    blocks = []
    inside_block = False
    lines = string.split("\n")
    for line in lines:
        if line[:len(start)] == start:
            inside_block = True
            blocks.append("")
        elif line[:len(end)] == end:
            inside_block = False
        elif inside_block:
            blocks[-1] += line + "\n"
    if return_list:
        return blocks
    return "\n".join(blocks)


if __name__ == "__main__":
    with open("./src/langprop/templates/example.txt", "r") as f:
        template = f.read()

    results = parse_template(template, people=["Tom", "Jerry"])
    print(results)
