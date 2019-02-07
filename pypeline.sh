#!/bin/sh

# #############################################################################
# pypeline.sh
# ===========
# Author : Sepand KASHANI [sep@zurich.ibm.com]
# #############################################################################

# Setup Pypeline environment + launch shell.

abs_script_dir="$(echo "${PWD}" | python3 -c "
import pathlib;
import sys;

file_path = sys.stdin.readline();
abs_cwd = pathlib.Path(file_path).absolute();

print(abs_cwd);
")"

show_help_message() {
    cat << EOF
usage: [source] pypeline.sh [-h] [--no_shell]

Launch Python3 interactive shell with Pypeline installed.

optional arguments:
  --no_shell (w/ source)   Only load Pypeline environment in current shell process.
                           This option is mainly used for development.

Environment variables in this script can be modified by the user to tailor Pypeline to their environment.

Warning
-------
This script must be executed/sourced from the directory where it is defined.

Examples
--------
sh     pypeline.sh             # Launch Pypeline shell.
source pypeline.sh --no_shell  # Just setup Pypeline environment.
EOF
}

has_conda() {
    which conda >& /dev/null
    return "${?}"
}

has_pypeline_env() {
    has_conda
    if [ "${?}" -eq 0 ]; then
        conda info -e | grep -e'pypeline_dev ' > /dev/null
        return "${?}"
    fi
}

load_pypeline_env() {
    source activate pypeline_dev

    # CasaCore: add <miniconda_root>/lib/ to LD_LIBRARY_PATH for libtinfow.so
    local miniconda_root="$(dirname "$(dirname "$(which conda)")")"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${miniconda_root}/lib"
}

if [ "${#}" -eq 1 ] && [ "${1}" = '-h' ]; then
    show_help_message
else
    has_conda
    if [ "${?}" -eq 0 ]; then
        if [ "${#}" -eq 1 ] && [ "${1}" = '--no_shell' ]; then
            has_pypeline_env
            if [ "${?}" -eq 0 ]; then
                load_pypeline_env
            else
                echo 'Error: pypeline_dev environment does not exist.'
            fi
        elif [ "${#}" -eq 0 ]; then
            has_pypeline_env
            if [ "${?}" -eq 0 ]; then
                load_pypeline_env
                ipython3 --matplotlib
            else
                echo 'Error: pypeline_dev environment does not exist.'
            fi
        else
            echo 'INCORRECT INVOCATION'
            echo
            show_help_message
        fi
    else
        echo "Error: could not locate conda executable."
    fi
fi
