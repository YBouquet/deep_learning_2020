"""
@author: Yann BOUQUET
"""

import os.path
import pandas as pd
from datetime import datetime as dt, date as d
import re
import argparse
import base64

from saver import DATETIME_FMT

#HEADER = ['datetime', 'target', 'model', 'summary', 'optimizer', 'learning_rate', 'epochs', 'minibatch_size', 'accuracy', 'f1_score', 'roc', 'b64_comments']

def check_date_segment(date, min_date, max_date):
    return date <= max_date and date >= min_date

def __search_by_date__(dataframe, year, month, day, hour = 0, minutes= 0):

    if hour == 0 and minutes == 0:
        current_date = d(year, month, day)
        result = dataframe[dataframe.datetime.apply(lambda x : current_date == dt.strptime(x.split(' ')[0], DATETIME_FMT.split(' ')[0]).date())]
    else:
        if minutes == 0:
            min_date, max_date = dt(year, month, day, hour = hour), dt(year, month, day, hour = hour, minute = 59)
        else:
            min_date, max_date = dt(year, month, day, hour = hour, minute= minutes), dt(year, month, day, hour = hour, minute = minutes, second = 59)
        result = dataframe[dataframe.datetime.apply(lambda x: check_date_segment(dt.strptime(x, DATETIME_FMT), min_date, max_date))]
    return result

def __search_from__(dataframe, year, month, day, hour = 0, minutes= 0):
    if hour == 0 and minutes == 0:
        min_date = d(year, month, day)
        result = dataframe[dataframe.datetime.apply(lambda x : min_date <= dt.strptime(x.split(' ')[0], DATETIME_FMT.split(' ')[0]).date())]
    else:
        min_datetime= dt(year, month, day, hour = hour, minute= minutes)
        result = dataframe[dataframe.datetime.apply(lambda x: dt.strptime(x, DATETIME_FMT) >= min_datetime)]
    return result
def __search_until__(dataframe, year, month, day, hour = 0, minutes= 0):
    if hour == 0 and minutes == 0:
        max_date = d(year, month, day)
        result = dataframe[dataframe.datetime.apply(lambda x : max_date >= dt.strptime(x.split(' ')[0], DATETIME_FMT.split(' ')[0]).date())]
    else:
        max_datetime= dt(year, month, day, hour = hour, minute= minutes)
        result = dataframe[dataframe.datetime.apply(lambda x: dt.strptime(x, DATETIME_FMT) <= max_datetime)]
    return result

def __search_by_model__(dataframe, model):
    return dataframe[dataframe.model.apply(lambda x: x.lower() == model.lower())]


def __search_by_target__(dataframe, target):
    return dataframe[dataframe.target.apply(lambda x: x.lower() == target.lower())]

def print_row(row):
    print('ID = ' + str(row.name) + ', Date: ' + row.datetime + ', Model: ' + row.model + \
                ', accuracy = ' + str(row.accuracy) +', f1 score = ' + str(row.f1_score) + ', roc = ' + str(row.roc) + \
                ', Comments: ' + base64.b64decode(str(row.b64_comments)).decode("utf-8") +'\n')

def print_result(dataframe):
    dataframe.sort_values(by=['accuracy']).apply(lambda row : print_row(row), axis = 1)

def search(filename, dates = None,  model = None, target = None):
    if not(os.path.isfile(filename)):
        print('ERROR : the file doesn\'t exist')
    else:
        report_df = pd.read_csv(filename, delimiter = ',')
        result = pd.DataFrame()
        if dates and len(dates) > 0 :
            if len(dates) != 2:
                for m_date in dates:
                    result = result.append(__search_by_date__(report_df, m_date.year, m_date.month, m_date.day, m_date.hour, m_date.minute))
            else:
                if dates[0] >= dates[1]:
                    max_date, min_date = dates[0], dates[1]
                else:
                    max_date, min_date = dates[1], dates[0]
                result = __search_from__(report_df, min_date.year, min_date.month, min_date.day, min_date.hour, min_date.minute)
                result = __search_until__(result, max_date.year, max_date.month, max_date.day, max_date.hour, max_date.minute)
        else:
            result = report_df.copy()

        if model:
            result = __search_by_model__(result, model)
        if target:
            result = __search_by_target__(result, target)

        if result.empty:
            print("No result")
        else:
            print_result(result.drop_duplicates())
        answer_dict = {'y':True, 'n':False}
        while True:
            try:
                # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
                inpt = input("Please enter the id of the test you want the details of: ")
                idx = int(inpt)
                try:
                    row = result.loc[idx]
                    print("SUMMARY OF THE ARCHITECTURE ------------------------------------\n")
                    print(base64.b64decode(row.summary).decode('utf-8'))
                    print("HYPERPARAMETERS ------------------------------------------------\n")
                    print("Optimizer: \t"+ row.optimizer + "\t|| Learning rate: \t"+str(row.learning_rate)+'\n')
                    print("Epochs: \t"+str(row.epochs)+'\t|| Minibatch size: \t'+str(row.minibatch_size)+'\n')
                except KeyError:
                    print('Please choose a valid id !')
                    continue
                on_going = True
                while True:
                    bool = input("Continue? [Y/n]: ")
                    if bool:
                        try:
                            on_going = answer_dict[bool.lower()]
                        except KeyError:
                            print("SERIOUSLY ?!")
                            continue
                    break
                if on_going:
                    continue
                else:
                    break
            except ValueError:
                if inpt == 'q':
                    break
                else :
                    print("The id is an integer !!!")
                    continue
def main(args):
    pd.set_option('display.expand_frame_repr', False)
    str_dates = args.dates.split(',')
    result = []
    for date in str_dates:
        m_date = re.findall('[0-9]*[/-][0-9]*[/-][0-9]*$', date)
        m_datetime = re.findall('[0-9]*[/-][0-9]*[/-][0-9]* [0-9]*:*[0-9]*$', date)
        if m_date:
            try:
                date_format = '%d-%m-%Y'
                for d_ in m_date:
                    if re.search('/', d_):
                        date_format = date_format.replace('-','/')
                    else:
                        date_format = date_format.replace('/','-')
                    result.append(dt.strptime(d_, date_format))
            except ValueError:
                print('ERROR : please reconsider the date format (DD/MM/YYYY or DD-MM-YYYY)')
        if m_datetime:
            try:
                datetime_format = '%d-%m-%Y %H'
                for d_ in m_datetime:
                    if re.search(':', d_):
                        datetime_format +=':%M'
                    if re.search('/', d_):
                        datetime_format = datetime_format.replace('-','/')
                    else:
                        datetime_format = datetime_format.replace('/','-')
                    result.append(dt.strptime(d_, datetime_format))
            except ValueError:
                print('ERROR : please reconsider the date format (DD/MM/YYYY or DD-MM-YYYY) and time format (hh or hh:mm)')
    search(args.filename, dates = result, model = args.model, target = args.target)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Searching elements in the report csv file')
    parser.add_argument('--filename', type = str,  help = 'name of the file where we should search for infos')
    parser.add_argument('--dates', type = str, default = '', help = 'date format : DD-MM-YYY ; time format : hh:mm; overall format : date{1} time{0,1}, date{1} time{0,1}, ...')
    parser.add_argument('--model', type = str, default = None)
    parser.add_argument('--target', type = str, default = None)

    args = parser.parse_args()
    if args.filename:
        main(args)
    else:
        print('ERROR: you have to give a file name')
