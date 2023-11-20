import logging, os, sys, gc, ctypes, torch



def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()
    print(f"memory on device 0 = {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    print(f"memory on device 1 = {torch.cuda.memory_allocated(1) / 1024 ** 3:.2f} GB")


def set_handlers(logger: logging.Logger, sys_id: int, root_dir:os.PathLike):
    # this function sets handlers for the logger
    logger.root.handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(root_dir / f"results/logs/id{sys_id}.log"),
    ]
    # add format to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    )
    for handler in logger.root.handlers:
        handler.setFormatter(formatter)
    logger.root.handlers
    logger.root.setLevel(logging.INFO)
    logger.info(f"sys id {sys_id}")
    return logger


def is_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # Not a Jupyter environment
            return False
    except (ImportError, AttributeError):
        return False

    return True