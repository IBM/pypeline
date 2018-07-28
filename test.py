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
cmds = dict(pytest=[f'source "{project_root_dir}/pypeline.sh" --no_shell',
                    f'pytest "{project_root_dir}/pypeline/test"'],
            flake8=[f'source "{project_root_dir}/pypeline.sh" --no_shell',
                    f'flake8 --ignore=E122,E128,E501,E731,E741 "{project_root_dir}/pypeline"'],
            doctest=[f'source "{project_root_dir}/pypeline.sh" --no_shell',
                     f'sphinx-build -b doctest "{project_root_dir}/doc" "{project_root_dir}/build/doctest"'])
for k in cmds:
    cmds[k].insert(0, 'export PYPELINE_RUNNING_TESTS=1')


parser = argparse.ArgumentParser(description='Pypeline test runner.',
                                 epilog='When run with no arguments, all tests are executed.')
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
    """
    if not isinstance(test_name, str):
        raise ValueError('Parameter[test_name] must be a str.')
    if test_name not in cmds:
        raise ValueError('Parameter[test_name] is not a valid test.')

    status = subprocess.run('; '.join(cmds[test_name]),
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
        print('Success' if (s.returncode == 0) else 'Failure')
        cmd_list = s.args.split('; ')
        for c in cmd_list:
            print(f'   {c}')
