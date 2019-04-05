import numpy as np
import pandas
import contextlib
import os
import joblib
import io
import time
from tqdm import tqdm

import logging # For Logs
logger = logging.getLogger(name=__name__)

def script_builder(
        script_name,
        python_script_path,
        python_script_args,
        path_to_shell_script,
        project_root,
        deploy_target="desktop",
        script_splits=1,
        conda_env_name=None,
        **kwargs,
        ):
    """ Create Shell Scripts for running experiments

    Args:
        script_name (string): name of script / experiment
        python_script_path (path): path to python script
        python_script_args (list of dicts): to pass to python script
        path_to_shell_script (path): path to shell script folder
        project_root (path): path to project root
        deploy_target (string): desktop
        script_splits (int): number of splits for desktop
        conda_env_name (string): conda virtual env to activate

    Details:
        Saves generated scripts into `<path_to_shell_script>/script_name`
        Logs are saved to `<path_to_shell_script>/logs`

    """
    logger.info("Setting Up Script Files for {0}".format(script_name))
    path_to_shell_script = os.path.join(path_to_shell_script, script_name)

    # Create Directories
    path_to_logs = os.path.join(path_to_shell_script,
           "{0}_logs".format(script_name))
    make_path(path_to_logs)

    logger.info("Setting up {0} experiments...".format(len(python_script_args)))
    if deploy_target == "desktop":
        bash_file_masters = create_desktop_jobs(
                list_of_args=python_script_args,
                script_name=script_name,
                python_script_path=python_script_path,
                path_to_shell_script=path_to_shell_script,
                path_to_logs=path_to_logs,
                project_root=project_root,
                conda_env_name=conda_env_name,
                script_splits=script_splits,
                )
    else:
        raise ValueError("Unrecognized deploy_target {0}".format(deploy_target))
    return bash_file_masters

def create_desktop_jobs(
        list_of_args,
        script_name,
        python_script_path,
        path_to_shell_script,
        path_to_logs,
        project_root,
        conda_env_name,
        script_splits,
        ):
    bash_file_masters=[None]*script_splits
    for split in range(script_splits):
        if script_splits > 1:
            bash_file_master = os.path.join(path_to_shell_script,
                "{0}_script_{1}.sh".format(script_name, split))
        else:
            bash_file_master = os.path.join(path_to_shell_script,
                "{0}_script.sh".format(script_name))
        # Write Shell Script
        with open(bash_file_master, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            f.write("cd " + project_root + "\n")
            f.write("\n")
            if conda_env_name is not None:
                f.write("source activate {0}\n".format(conda_env_name))

            for ii, args in enumerate(list_of_args):
                if (ii >= split*len(list_of_args)/script_splits) and \
                        (ii < (split+1)*len(list_of_args)/script_splits):
                    log_name = "{0:0>3}.log".format(
                            args.get('--experiment_id', 0))
                    log_file_name = os.path.join(path_to_logs, log_name)
                    python_args = ops_dict_to_string_args(args)

                    f.write("\n")
                    f.write("python " + python_script_path + " " +
                            python_args)
                    f.write(" |& tee " + log_file_name + ".out\n")
        os.chmod(bash_file_master, 0o755)
        bash_file_masters[split] = bash_file_master

    return bash_file_masters

def make_path(path):
    # Helper function for making directories
    if path is not None:
        if not os.path.isdir(path):
            if os.path.exists(path):
                raise ValueError(
        "path {0} is any existing file location!".format(path)
                            )
            else:
                # To avoid race conditions
                wait_time = np.random.rand()*2
                logging.info("Pausing for {0:2.2f} sec to make {1}".format(
                    wait_time, path))
                time.sleep(wait_time)
                if os.path.isdir(path):
                    return

                # Make Dirs
                try:
                    os.makedirs(path)
                except OSError as e:
                    logger.error(e.strerror)
                    import errno
                    if e.errno == errno.EEXIST:
                        logger.info("Ignoring Race Condition Error")
                        pass
                    else:
                        raise e
    return

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.

        From https://github.com/tqdm/tqdm/issues/313
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

def ops_dict_to_string_args(arg_dict, sep=" "):
    string_args = ""
    for k, v in arg_dict.items():
        if type(v) is str:
            string_args += k + sep + '"' + str(v) + '"' + sep
        elif v is None:
            string_args += k + sep
        else:
            string_args += k + sep + str(v) + sep
    return(string_args)

## Context Writers

# See https://stackoverflow.com/questions/42409707/pandas-to-csv-overwriting-prevent-data-loss?rq=1 for explanation on why we use context managers

@contextlib.contextmanager
def atomic_overwrite(filename):
    temp = filename + "~"
    with open(temp, "w") as f:
        yield f
    os.rename(temp, filename)

def pandas_write_df_to_csv(df, filename, **kwargs):
    """ Write DF to filename (via temp file)
    Optional kwargs:
        index = False
    Note a better solution is to use mode='a'
    """
    with atomic_overwrite(filename) as f:
        df.to_csv(f, **kwargs)
    return

@contextlib.contextmanager
def atomic_overwrite_binary(filename):
    temp = filename + "~"
    with open(temp, "wb") as f:
        yield f
    os.rename(temp, filename)

def joblib_write_to_file(data, filename, **kwargs):
    with atomic_overwrite_binary(filename) as f:
        joblib.dump(data, f, **kwargs)
    return

#
