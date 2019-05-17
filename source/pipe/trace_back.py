import functools
import traceback


def process_traceback(f):
    '''
    This decorator will help debug easier in multiprocess environment
    '''

    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise e

    return func
