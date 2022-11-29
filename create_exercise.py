'''
The script replaces all lines
marked at the line end with: # REMOVE
and replaces it with: #your solution

Run:
python ./create_exercise.py --ifiles solution_preprocessing.ipynb solution_linear_regression ...

TODO: .ipynb are json files, therefore reimplement using json directly!
'''


import argparse
from pathlib import Path
import re


def create_exercise(lines):
    out = []

    for line in lines:
        # match the pattern: "# REMOVE"
        pattern = r'^(\s*")(\s*)(.*)(#\s*REMOVE)(.*",?\s*)'
        if match := re.search(pattern, line):

            # replace it with: "#your solution"
            fill = re.sub(pattern, r"\1\2##your solution\5", line)
            out.append(fill)
        else:
            out.append(line)

        # match single closing bracket: ']'
        pattern = '^\s*]\s*$'
        if re.search(pattern, line):

            # correct for surplus: ","
            pattern = '(.*)(,)(\s*$)'
            if match := re.search(pattern, out[-2]):
                out[-2] = match.group(1) + match.group(3)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ifiles', nargs='+')
    args = parser.parse_args()

    for fname in args.ifiles:

        ifile = Path(fname)
        with ifile.open() as f:

            # read file into lines
            lines = f.readlines()

            # apply new line with create_exercise
            new_lines = create_exercise(lines)

        # write file
        name = ['exercise'] + str(ifile.name).split('_')[1:]
        ofile = Path(ifile.parents[0] / '_'.join(name))
        with ofile.open('w') as f:
            for line in new_lines:
                f.write(line)
