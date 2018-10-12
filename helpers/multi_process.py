from multiprocessing import Pool, cpu_count
import signal


def map_in_pool(fn, data, single_process=False, verbose=False, nr_pro=4):
    """
    Our multiprocessing solution; wrapped to stop on ctrl-C well.
    """
    if single_process:
        return map(fn, data)
    n_procs = min(cpu_count(), nr_pro)
    original_sigint_handler = setup_sigint()
    pool = Pool(processes=n_procs, initializer=setup_sigint)
    restore_sigint(original_sigint_handler)
    try:
        if verbose:
            print('Mapping with %d processes' % n_procs)
        res = pool.map_async(fn, data)
        return res.get(31536000)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()


def setup_sigint():
    return signal.signal(signal.SIGINT, signal.SIG_IGN)


def restore_sigint(original):
    signal.signal(signal.SIGINT, original)