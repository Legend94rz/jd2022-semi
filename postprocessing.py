import argparse
import json
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, nargs='+')
    parser.add_argument('-o', '--output_file', type=str, default='')
    parser.add_argument('-r', '--will_reverse', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_inputs = []
    file_name = []
    for input_file in args.input_file:
        input_file = Path(input_file)
        with open(input_file, 'r') as fin:
            objs = [json.loads(x) for x in fin.readlines()]
        all_inputs.append(objs)
        file_name.append(input_file.stem)

    if not args.output_file:
        output_file = '+'.join(file_name) + f'-final.txt'
    else:
        output_file = args.output_file
    with open(output_file, 'w') as fout:
        for i, o in enumerate(zip(*all_inputs)):
            assert all(a['img_name']==o[0]['img_name'] for a in o) and all(set(a['match']) == set(o[0]['match']) for a in o)
            x = {
                'img_name': o[0]['img_name'],
            }
            score = {}
            for k in o[0]['match']:
                v = np.mean([a['match'][k] for a in o])
                score[k] = (1 - int(v>0.5)) if args.will_reverse else int(v>0.5)
            x['match'] = score
            if x['match']['图文']:
                if any(not a for a in x['match'].values()):
                    print(f"warning: {x['img_name']} match, but one attr is not match: {x['match']}")
            fout.write(json.dumps(x, ensure_ascii=False) + '\n')
    print(f'writen submits to {output_file}!')
