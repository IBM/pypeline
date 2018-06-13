#!/usr/bin/env python3

# #############################################################################
# test.py
# =======
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Pypeline test script.
"""

import argparse
import pathlib
import subprocess
import sys

project_root_dir = pathlib.Path(__file__).parent.absolute()
cmds = dict(pytest=[(f'pytest '
                     f'{project_root_dir}/pypeline/test'), ],
            flake8=[(f'flake8 '
                     f'--ignore=E122,E128,E501,E731,E741 '
                     f'{project_root_dir}/pypeline'), ],
            doctest=[(f'sphinx-build '
                      f'-b '
                      f'doctest '
                      f'{project_root_dir}/doc/source '
                      f'{project_root_dir}/doc/build/doctest'), ], )

parser = argparse.ArgumentParser(description='Pypeline test runner.',
                                 epilog=(
                                     'When run with no arguments, all tests '
                                     'are executed.'))
parser.add_argument('-e',
                    help='Name of test to run.',
                    type=str,
                    choices=cmds.keys())
args = parser.parse_args()


def run_test(test_name):
    """
    Execute a test on the command line.

    Parameters
    ----------
    test_name : str
        Name of a key in `cmds`.

    Raises an exception if the test fails.
    """
    if not isinstance(test_name, str):
        raise ValueError('Parameter[test_name] must be a str.')
    if test_name not in cmds:
        raise ValueError('Parameter[test_name] is not a valid test.')

    command_list = cmds[test_name]
    for command in command_list:
        print(f' Executing {command}.')
        status = subprocess.run(command,
                                stdin=None,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                shell=True)
        return status


if __name__ == '__main__':
    status = []

    if args.e is None:
        for test in cmds:
            s = run_test(test)
            status.append(s)
    else:
        s = run_test(args.e)
        status.append(s)

    print('\nSummary\n=======')
    for s in status:
        p = 'Success' if (s.returncode == 0) else 'Failure'
        c = s.args
        print(f'{p} : {c}.')
