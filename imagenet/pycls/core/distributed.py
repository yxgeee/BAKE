#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed helpers."""

import multiprocessing
import os
import subprocess
import random
import signal
import threading
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pycls.core.config import cfg


def is_master_proc():
    """Determines if the current process is the master process.

    Master process is responsible for logging, writing and loading checkpoints. In
    the multi GPU setting, we assign the master role to the rank 0 process. When
    training using a single GPU, there is a single process which is considered master.
    """
    return cfg.NUM_GPUS == 1 or torch.distributed.get_rank() == 0


def init_process_group(proc_rank, world_size, port):
    """Initializes the default process group."""
    # Set the GPU to use
    torch.cuda.set_device(proc_rank)
    # Initialize the process group
    torch.distributed.init_process_group(
        backend=cfg.DIST_BACKEND,
        init_method="tcp://{}:{}".format(cfg.HOST, port),
        world_size=world_size,
        rank=proc_rank,
    )


def destroy_process_group():
    """Destroys the default process group."""
    torch.distributed.destroy_process_group()


def scaled_all_reduce(tensors):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group (equivalent to cfg.NUM_GPUS).
    """
    # There is no need for reduction in the single-proc case
    if cfg.NUM_GPUS == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / cfg.NUM_GPUS)
    return tensors


class ChildException(Exception):
    """Wraps an exception from a child process."""

    def __init__(self, child_trace):
        super(ChildException, self).__init__(child_trace)


class ErrorHandler(object):
    """Multiprocessing error handler (based on fairseq's).

    Listens for errors in child processes and propagates the tracebacks to the parent.
    """

    def __init__(self, error_queue):
        # Shared error queue
        self.error_queue = error_queue
        # Children processes sharing the error queue
        self.children_pids = []
        # Start a thread listening to errors
        self.error_listener = threading.Thread(target=self.listen, daemon=True)
        self.error_listener.start()
        # Register the signal handler
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """Registers a child process."""
        self.children_pids.append(pid)

    def listen(self):
        """Listens for errors in the error queue."""
        # Wait until there is an error in the queue
        child_trace = self.error_queue.get()
        # Put the error back for the signal handler
        self.error_queue.put(child_trace)
        # Invoke the signal handler
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, _sig_num, _stack_frame):
        """Signal handler."""
        # Kill children processes
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)
        # Propagate the error from the child process
        raise ChildException(self.error_queue.get())


def run(proc_rank, world_size, port, error_queue, fun, fun_args, fun_kwargs):
    """Runs a function from a child process."""
    try:
        # Initialize the process group
        init_process_group(proc_rank, world_size, port)
        # Run the function
        fun(*fun_args, **fun_kwargs)
    except KeyboardInterrupt:
        # Killed by the parent process
        pass
    except Exception:
        # Propagate exception to the parent process
        error_queue.put(traceback.format_exc())
    finally:
        # Destroy the process group
        destroy_process_group()

def multi_proc_run(num_proc, fun, fun_args=(), fun_kwargs=None):
    init_dist()
    cfg.freeze()
    fun_kwargs = fun_kwargs if fun_kwargs else {}
    fun(*fun_args, **fun_kwargs)
    # return

def init_dist(backend="nccl"):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    if cfg.LAUNCHER == "pytorch":
        init_dist_pytorch(backend)
    elif cfg.LAUNCHER == "slurm":
        init_dist_slurm(backend)
    else:
        raise ValueError("Invalid launcher type: {}".format(cfg.LAUNCHER))


def init_dist_pytorch(backend="nccl"):
    cfg.RANK = int(os.environ["LOCAL_RANK"])
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        cfg.NGPUS_PER_NODE = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        cfg.NGPUS_PER_NODE = torch.cuda.device_count()
    assert cfg.NGPUS_PER_NODE>0, "CUDA is not supported"
    cfg.GPU = cfg.RANK
    torch.cuda.set_device(cfg.GPU)
    dist.init_process_group(backend=backend)
    cfg.NUM_GPUS = dist.get_world_size()
    cfg.WORLD_SIZE = cfg.NUM_GPUS

def init_dist_slurm(backend="nccl"):
    cfg.RANK = int(os.environ["SLURM_PROCID"])
    cfg.WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        cfg.NGPUS_PER_NODE = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        cfg.NGPUS_PER_NODE = torch.cuda.device_count()
    assert cfg.NGPUS_PER_NODE>0, "CUDA is not supported"
    cfg.GPU = cfg.RANK % cfg.NGPUS_PER_NODE
    torch.cuda.set_device(cfg.GPU)
    addr = subprocess.getoutput(
        "scontrol show hostname {} | head -n1".format(node_list)
    )
    os.environ["MASTER_PORT"] = str(cfg.PORT)
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(cfg.WORLD_SIZE)
    os.environ["RANK"] = str(cfg.RANK)
    dist.init_process_group(backend=backend)
    cfg.NUM_GPUS = dist.get_world_size()
