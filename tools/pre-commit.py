#!/usr/bin/env python3

# #############################################################################
# pre-commit.py
# =============
# Git pre-commit hook.
#
# Author : Sahand Kashani-Akhavan [sahand.kashani-akhavan@epfl.ch]
# Revision : 0.10
# Last updated : 2018-04-06 08:19:25 UTC
# #############################################################################

"""
Git pre-commit hook that updates revision/last-updated fields in staged document
headers.
"""

import datetime
import itertools
import re
import subprocess
import sys


# #############################################################################
# Helpers #####################################################################
# #############################################################################
def parse_git_files(cmd):
    """
    Returns the list of files which git displays on stdout when queried with
    the command.

    :param cmd: List[str] describing the command.
                Example: ["git", "diff", "--cached"].
    :return: List[str] where each string is the relative path of a file from
             the root of the git repository.
    """
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = str(proc.stdout, 'utf-8').strip()
    files = output.split()
    return files


def file_data(f, commit=""):
    """
    Returns the contents of the file as a single large string.

    :param f: Input file name
    :param commit: Index entry (examples: "HEAD", "6af019e"). You can use the
                   empty string "" to specify the current working directory as
                   the "commit".
    :return: Contents of file as a single large string.
    """
    cmd = ["git", "--no-pager", "show", commit + ":" + f]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    return str(proc.stdout, 'utf-8')


# #############################################################################
# Git commands ################################################################
# #############################################################################
def git_staged_renamed_file_list():
    """
    Returns list of files which have been renamed in the git repository (the
    file name could have changed, but content is 100% identical).

    :return: Dict[str -> str] where each string is the relative path of a file
             from the root of the git repository. For each mapping, the key
             corresponds to the old location of a file, and the value
             corresponds to the new location of the same file.
    """
    # List of files which have been added.
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=A"]
    added_files = parse_git_files(cmd)

    # List of files which have been removed.
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=D"]
    removed_files = parse_git_files(cmd)

    renamed_files = dict()

    for (f_added, f_removed) in itertools.product(added_files, removed_files):
        f_added_data = file_data(f_added)

        # The removed file can no longer be found in the current working
        # directory, so must explicitly name commit to search in.
        f_removed_data = file_data(f_removed, "HEAD")

        if f_added_data == f_removed_data:
            renamed_files[f_removed] = f_added
    return renamed_files


def git_staged_added_file_list():
    """
    Returns list of files which have been added to the git repository. Renamed
    files (just "moved", but with identical file contents to their original
    location) are excluded from the results.

    :return: List[str] where each string is the relative path of a file from
             the root of the git repository.
    """
    # List of files which have been added (includes files which have been
    # renamed).
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=A"]
    all_added_files = parse_git_files(cmd)

    # List of renamed files with 100% identical content before and after
    # renaming.
    renamed_files = git_staged_renamed_file_list()

    # List of files which have been added (excludes files which have been
    # renamed).
    added_files = set(all_added_files) - set(renamed_files.values())
    return list(added_files)


def git_staged_modified_file_list():
    """
    Returns list of files which have been modified in the git repository.

    :return: List[str] where each string is the relative path of a file from
             the root of the git repository.
    """
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=M"]
    return parse_git_files(cmd)


def git_save_stash():
    """
    Stashes all unstaged changes in the git repository.

    :return: None
    """
    cmd = ["git", "stash", "-q", "--keep-index"]
    subprocess.run(cmd)


def git_pop_stash():
    """
    Pops the git stash.

    :return: None
    """
    cmd = ["git", "stash", "pop", "-q"]
    subprocess.run(cmd)


def git_stage(f):
    """
    Stages the input file in the git repository.

    :param f: Input file name.
    :return: None
    """
    cmd = ["git", "add", "-u", f]
    subprocess.run(cmd)


# #############################################################################
# File transformations ########################################################
# #############################################################################
def update_revision(f):
    """
    Updates the revision of the file by incrementing its minor number. Note
    that revisions must satisfy the following pattern in order to be updated
    (shown in regular expression form):

        "Revision\s*:\s*(?P<major>\d+)\.(?P<minor>\d+)"

    If no revision satisfying the above pattern is found in the file, then
    the file is left untouched.

    :param f: Input file name.
    :return: None
    """
    with open(f, "r+") as fp:
        # Read complete file
        data = fp.read()

        # Search for revision
        pattern = r"Revision\s*:\s*(?P<major>\d+)\.(?P<minor>\d+)"
        match = re.search(pattern, data)

        if match:
            # Increment revision
            new_major = int(match.group("major"))
            new_minor = int(match.group("minor")) + 1
            replacement = f"Revision : {new_major}.{new_minor}"
            data = re.sub(pattern, replacement, data)

        # Overwrite old file contents
        fp.seek(0)
        fp.write(data)
        fp.truncate()


def update_timestamp(f):
    """
    Updates the timestamp of the file. Note that timestamps must be in UTC
    format and must satisfy the following pattern in order to be updated
    (shown in regular expression form):

        (r"Last updated\s*:\s*"
         r"(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+) "
         r"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+) UTC")

    If no timestamp satisfying the above pattern is found in the file, then
    the file is left untouched.

    :param f: Input file name.
    :return: None
    """
    with open(f, "r+") as fp:
        # Read complete file
        data = fp.read()

        # Search for timestamp
        # datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        pattern = (r"Last updated\s*:\s*"
                   r"(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+) "
                   r"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+) UTC")
        match = re.search(pattern, data)

        if match:
            # Increment timestamp
            utctime = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            replacement = f"Last updated : {utctime} UTC"
            data = re.sub(pattern, replacement, data)

        # Overwrite old file contents
        fp.seek(0)
        fp.write(data)
        fp.truncate()


# Main program ----------------------------------------------------------------
# git_save_stash()
added_files = git_staged_added_file_list()
for f in added_files:
    update_timestamp(f)
    git_stage(f)

modified_files = git_staged_modified_file_list()
for f in modified_files:
    update_revision(f)
    update_timestamp(f)
    git_stage(f)
# git_pop_stash()

sys.exit(0)
