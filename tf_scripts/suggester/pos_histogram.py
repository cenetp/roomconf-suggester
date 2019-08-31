import json
import operator
import os
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser()
parser.add_argument("-a", "--action", dest="action", required=False)
parser.add_argument("-n", "--number", dest="number", required=False)
args = parser.parse_args()


def create_json():
    if not os.path.exists(os.getcwd() + '/tf_scripts/suggester/cases_csv/positions.json'):
        f = open(os.getcwd() + '/tf_scripts/suggester/cases_csv/positions.txt')
        lines = f.readlines()

        combinations_full = []
        for i in range(len(lines)):
            combi = str(lines[i]).split('-')
            if len(combi) == 2 and combi[1] != '\n' and not combi[1].startswith('('):
                combinations_full.append(combi[0] + ',' + combi[1].replace('\n', ''))

        unique_combis = Counter(combinations_full).keys()
        unique_combi_count = Counter(combinations_full).values()

        # print(unique_combis)
        # print(unique_combi_count)
        # print(len(unique_combis))
        # print(len(unique_combi_count))

        keys_list = []
        keys_no_pos_list = []
        for key in unique_combis:
            keys_list.append(key)
            keys_no_pos_list.append(key.split(',')[0])

        values_list = []
        for value in unique_combi_count:
            values_list.append(value)

        unique_actions = Counter(keys_no_pos_list).keys()
        union_dict = {}
        for key in keys_no_pos_list:
            union_dict[key] = []

        for i in range(len(keys_list)):
            union_dict[keys_list[i].split(',')[0]].append((keys_list[i].split(',')[1], values_list[i]))

        for key in union_dict:
            union_dict[key].sort(key=operator.itemgetter(1), reverse=True)

        ff = open(os.getcwd() + '/tf_scripts/suggester/cases_csv/positions.json', 'w')
        ff.write(json.dumps(union_dict, indent=2))


def suggest_connection(action, number):
    with open(os.getcwd() + '/tf_scripts/suggester/cases_csv/positions.json') as jsn:
        conns = json.load(jsn)
        try:
            action_conns = list(conns[action])
            print(action_conns[int(number)][0])
            return action_conns[int(number)][0]
        except KeyError:
            print('0')
            return '0'
        except IndexError:
            print('0')
            return '0'


create_json()
suggest_connection(args.action, args.number)
