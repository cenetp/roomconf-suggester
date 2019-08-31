from argparse import ArgumentParser
import os, random
from os import listdir
from os.path import isfile, join

dirname = os.getcwd() + '/tf_scripts/suggester/cases_csv'
filenames = [n for n in listdir(dirname) if isfile(join(dirname, n))]

if not os.path.exists(os.getcwd() + '/tf_scripts/suggester/cases_csv/footprint_cases'):
    os.mkdir(os.getcwd() + '/tf_scripts/suggester/cases_csv/footprint_cases')

for filename in filenames:
    if filename.endswith('.csv'):
        footprint_filename = os.getcwd() + '/tf_scripts/suggester/cases_csv/footprint_cases/footprint_' + filename
        if os.path.exists(footprint_filename):
            os.remove(footprint_filename)
        file = open(os.getcwd() + '/tf_scripts/suggester/cases_csv/' + filename, 'r', encoding='utf-8')
        file_footprint = open(footprint_filename, 'a', encoding='utf-8')
        lines = file.readlines()
        # 5% of all cases will be used as footprint cases
        num_footprint_cases = round(len(lines) / 100 * 5)
        for i in range(num_footprint_cases):
            random_case_num = random.randrange(0, len(lines))
            random_case = lines[random_case_num].split(',')
            # create footprint case
            file_footprint.write(random_case[2] + ',' + random_case[3] + ',' + random_case[4] + ',' + random_case[5])
            lines.remove(lines[random_case_num])

print('Footprints created.')
