#!/usr/bin/env python3

# #############################################################################
# build.py
# ========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

"""
Pypeline build script.
"""

import argparse
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser(description='Install Pypeline tools.',
                                 epilog="""
Examples
--------
python3 pypeline_build --doc
python3 pypeline_build --lib=Debug
                       --C_compiler /usr/bin/clang
                       --CXX_compiler /usr/bin/clang++
                                 """,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--lib',
                   help='Compile C++/Python libraries in Debug or Release mode.',
                   type=str,
                   choices=['Debug', 'Release'])
group.add_argument('--doc',
                   help=('Generate HTML documentation. This should only be '
                         'used after an initial invocation of the script '
                         'using --lib.'),
                   action='store_true')
parser.add_argument('--C_compiler',
                    help='C compiler executable. Use system default if unspecified.',
                    type=str,
                    required=False)
parser.add_argument('--CXX_compiler',
                    help='C++ compiler executable. Use system default if unspecified.',
                    type=str,
                    required=False)
parser.add_argument('--OpenMP',
                    help='Use OpenMP',
                    action='store_true')
parser.add_argument('--print',
                    help=('Only print commands that would have been executed '
                          'given specified options.'),
                    action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    project_root_dir = pathlib.Path(__file__).parent.absolute()
    cmds = dict(doc=[f'source "{project_root_dir}/pypeline.sh" --no_shell',
                     f'python3 "{project_root_dir}/setup.py" build_sphinx'],
                lib=[f'source "{project_root_dir}/pypeline.sh" --no_shell',
                     f'PYPELINE_CPP_BUILD_DIR="{project_root_dir}/build/cpp"',
                     f'rm -rf "${{PYPELINE_CPP_BUILD_DIR}}"',
                     f'mkdir --parents "${{PYPELINE_CPP_BUILD_DIR}}"',
                     f'cd "${{PYPELINE_CPP_BUILD_DIR}}"',
                     (f'cmake -DCMAKE_BUILD_TYPE={args.lib} ' +
                      (f'-DCMAKE_C_COMPILER="{args.C_compiler}" ' if (args.C_compiler is not None) else '') +
                      (f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}" ' if (args.CXX_compiler is not None) else '') +
                      f'-DPYPELINE_USE_OPENMP={str(args.OpenMP).upper()} ' +
                      f'"{project_root_dir}"'),
                     'make',
                     f'cd "{project_root_dir}"',
                     f'python3 "{project_root_dir}/setup.py" develop'])

    if args.lib is not None:
        cmd_list = cmds['lib']
    else:
        cmd_list = cmds['doc']

    if args.print is True:
        print('\n'.join(cmd_list))
    else:
        status = subprocess.run('; '.join(cmd_list),
                                stdin=None,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                shell=True,
                                cwd=project_root_dir)

        print('\nSummary\n=======')
        print('Success' if (status.returncode == 0) else 'Failure')
        for c in cmd_list:
            print(f'   {c}')
