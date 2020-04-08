import os.path

import numpy as np
from datetime import datetime as dt

DATETIME_FMT = '%d-%m-%Y %H:%M:%S'
HEADER = np.array([['datetime', 'target', 'model', 'optimizer', 'epochs', 'minibatch_size', 'accuracy', 'f1_score', 'roc', 'comments']])


def save_csv(filename, infos):
    try :
        m_list =   np.array([
                                [
                                    dt.today().strftime(DATETIME_FMT),
                                    infos['target'],
                                    infos['model'],
                                    infos['optimizer'],
                                    infos['epochs'],
                                    infos['minibatch_size'],
                                    infos['accuracy'],
                                    infos['f1_score'],
                                    infos['roc'],
                                    infos['comments']
                                ]
                    ])
        if os.path.isfile(filename):
            f = open(filename, 'a')
            np.savetxt(f, m_list, fmt = '%s', delimiter = ',')
        else:
            f = open(filename, 'a+')
            np.savetxt(f, HEADER,  fmt = '%s', delimiter  = ',')
            np.savetxt(f, m_list, fmt = '%s', delimiter = ',')
        f.close()
    except KeyError:
        print('ERROR : the infos are not complete')
