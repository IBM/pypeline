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
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request

parser = argparse.ArgumentParser(description='Install Pypeline tools.',
                                 epilog="""
Examples
--------
python3 build.py --doc
python3 build.py --lib=Debug
                 --C_compiler /usr/bin/clang
                 --CXX_compiler /usr/bin/clang++
python3 build.py --install_dependencies
                 --C_compiler /usr/bin/gcc-7
                 --CXX_compiler /usr/bin/gcc-7
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
group.add_argument('--download_dependencies',
                   help='Download dependencies and extract archives.',
                   action='store_true')
group.add_argument('--install_dependencies',
                   help=('Install Pypeline\'s C++ dependencies. '
                         'This command must be used after calling --download_dependencies'),
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
    archive_dir = project_root_dir / 'dependencies'

    if args.lib is not None:
        build_dir = f'{project_root_dir}/build/pypeline'
        cmds = f'''
source "{project_root_dir}/pypeline.sh" --no_shell;
rm -rf "{build_dir}";
mkdir --parents "{build_dir}";
cd "{build_dir}";
cmake -DCMAKE_BUILD_TYPE="{args.lib}"              \
   {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
   {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
      -DPYPELINE_USE_OPENMP={str(args.OpenMP).upper()} \
      "{project_root_dir}";
make install;
cd "{project_root_dir}";
python3 "{project_root_dir}/setup.py" develop;
'''

    elif args.doc is True:
        cmds = f'''
source "{project_root_dir}/pypeline.sh" --no_shell;
python3 "{project_root_dir}/setup.py" build_sphinx;
'''

    elif args.download_dependencies is True:
        if not archive_dir.exists():
            archive_dir.mkdir(parents=True)

        for web_link in ['https://github.com/QuantStack/xtl/archive/0.4.15.tar.gz',
                         'https://github.com/QuantStack/xsimd/archive/6.1.6.tar.gz',
                         'https://github.com/QuantStack/xtensor/archive/0.17.3.tar.gz',
                         'http://bitbucket.org/eigen/eigen/get/3.3.5.tar.gz',
                         'https://github.com/pybind/pybind11/archive/v2.2.3.tar.gz',
                         'http://www.fftw.org/fftw-3.3.8.tar.gz']:
            print(f'Downloading {web_link}')
            with urllib.request.urlopen(web_link) as response:
                archive_path = archive_dir / os.path.basename(web_link)
                with archive_path.open(mode='wb') as archive:
                    shutil.copyfileobj(response, archive)

            with tarfile.open(archive_path) as archive:
                extracted_dir = archive_dir / os.path.commonprefix(archive.getnames())
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner) 
                    
                
                safe_extract(archive, path=extracted_dir)

        cmds = ''  # for '--print' to not fail if specified.

    elif args.install_dependencies is True:
        def find_extracted_folder(name):
            """
            Given the name of a dependency, find the directory in dependencies/ to which it was extracted.
            """
            dep_dir = project_root_dir / 'dependencies'
            candidates = [_ for _ in dep_dir.iterdir()
                          if _.is_dir() and (name in _.name)]
            if len(candidates) == 1:
                return candidates[0].absolute() / candidates[0].name
            else:
                raise ValueError(f'Could not locate directory containing "{name}".')

        def xtl():
            build_dir = project_root_dir / 'build' / 'xtl'
            extracted_dir = find_extracted_folder('xtl')

            cmds = f'''
mkdir -p "{build_dir}";
cd "{build_dir}";
cmake -DCMAKE_INSTALL_PREFIX="{project_root_dir}"  \
   {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
   {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
      "{extracted_dir}";
make install;
'''
            return cmds


        def xsimd():
            build_dir = project_root_dir / 'build' / 'xsimd'
            extracted_dir = find_extracted_folder('xsimd')

            cmds = f'''
mkdir -p "{build_dir}";
cd "{build_dir}";
cmake -DCMAKE_INSTALL_PREFIX="{project_root_dir}"  \
      -DBUILD_TESTS=OFF \
   {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
   {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
      "{extracted_dir}";
make install;
'''
            return cmds


        def xtensor():
            build_dir = project_root_dir / 'build' / 'xtensor'
            extracted_dir = find_extracted_folder('xtensor')

            cmds = f'''
mkdir -p "{build_dir}";
cd "{build_dir}";
cmake -DCMAKE_INSTALL_PREFIX="{project_root_dir}"  \
   {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
   {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
      "{extracted_dir}";
make install;
'''
            return cmds


        def eigen():
            build_dir = project_root_dir / 'build' / 'eigen'
            extracted_dir = find_extracted_folder('eigen')

            cmds = f'''
mkdir -p "{build_dir}";
cd "{build_dir}";
cmake -DCMAKE_INSTALL_PREFIX="{project_root_dir}"  \
      -DCMAKE_INSTALL_DATADIR="{project_root_dir}/lib64" \
   {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
   {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
      "{extracted_dir}";
make install;
'''
            return cmds


        def pybind11():
            build_dir = project_root_dir / 'build' / 'pybind11'
            extracted_dir = find_extracted_folder('pybind11')

            cmds = f'''
mkdir -p "{build_dir}";
cd "{build_dir}";
cmake -DCMAKE_INSTALL_PREFIX="{project_root_dir}"  \
      -DPYBIND11_INSTALL=ON \
      -DPYBIND11_TEST=OFF \
      -DPYBIND11_CMAKECONFIG_INSTALL_DIR="{project_root_dir}/lib64/cmake/pybind11" \
   {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
   {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
      "{extracted_dir}";
make install;
'''
            return cmds


        def fftw():
            build_dir = project_root_dir / 'build' / 'fftw'
            extracted_dir = find_extracted_folder('fftw')

            # FFTW's cmake interface cannot build float/double libraries at the same time, hence we have to relaunch the commands with 2 different values for ENABLE_FLOAT.
            gen_cmd = lambda compile_float: f'''
mkdir -p "{build_dir}";
cd "{build_dir}";
cmake -DCMAKE_INSTALL_PREFIX="{project_root_dir}"  \
      -DENABLE_OPENMP={str(args.OpenMP).upper()} \
      -DENABLE_THREADS={str(args.OpenMP).upper()} \
      -DENABLE_FLOAT={compile_float} \
      -DENABLE_SSE=ON \
      -DENABLE_SSE2=ON \
      -DENABLE_AVX=ON \
      -DENABLE_AVX2=ON \
      -DDISABLE_FORTRAN=ON \
   {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
   {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
      "{extracted_dir}";
make install;
'''
            return '\n'.join([gen_cmd('OFF'), gen_cmd('ON')])


        cmds = '\n'.join([f'source "{project_root_dir}/pypeline.sh" --no_shell;',
                          xtl(), xsimd(), xtensor(),
                          eigen(), pybind11(), fftw()])

    if args.print is True:
        print(cmds)
    else:
        status = subprocess.run(cmds,
                                stdin=None,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                shell=True,
                                cwd=project_root_dir)

        print('\nSummary\n=======')
        print('Success' if (status.returncode == 0) else 'Failure')
        print(cmds)
