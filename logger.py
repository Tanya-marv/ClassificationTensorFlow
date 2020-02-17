import logging
import datetime
import os

def initialize_log_file():
    dirName = 'logs'

    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    logging.basicConfig(filename=datetime.datetime.now()
                        .strftime("logs/log_%Y.%m.%d-%H:%M:%S.%f.txt"))
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def logging_level_verbosity(logging_verbosity):
    """Converts logging_level into TensorFlow logging verbosity value.

  Args:
    logging_verbosity: String value representing logging level: 'DEBUG', 'INFO',
    'WARN', 'ERROR', 'FATAL'
  """
    name_to_level = {
        'FATAL': logging.FATAL,
        'ERROR': logging.ERROR,
        'WARN': logging.WARN,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG
    }

    try:
        return name_to_level[logging_verbosity]
    except Exception as e:
        raise RuntimeError('Not supported logs verbosity (%s). Use one of %s.' %
                           (str(e), list(name_to_level)))