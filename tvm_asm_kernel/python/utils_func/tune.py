import tvm
from tvm import te
from tvm import autotvm

import os
from config.common_config import measure_option

def tune(
    M,
    K,
    N,
    record_file,
    parallel, 
    n_trial=2500,
    early_stopping=1000,
    instruction="neon", 
    target="llvm",
):
    best_record = record_file
    record_file += ".tmp"
    if os.path.exists(best_record):
        os.remove(best_record)
    if os.path.exists(record_file):
        os.remove(record_file)

    task = autotvm.task.create(
        "matmul", args=[M, K, N, parallel, instruction], target=target
    )
    print(task.config_space)

    # tuner = autotvm.tuner.XGBTuner(task)
    tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
    # tuner = autotvm.tuner.GridSearchTuner(task)

    tuner.tune(
        n_trial=n_trial,
        early_stopping=early_stopping,
        measure_option=measure_option,
        callbacks=[
            autotvm.callback.progress_bar(n_trial, prefix='\n'),
            autotvm.callback.log_to_file(record_file),
        ],
    )

    autotvm.record.pick_best(record_file, best_record)