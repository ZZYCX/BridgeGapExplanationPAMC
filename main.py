import os
import traceback
import sys
from config import get_configs
from train import run_train

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

def main():
    P = get_configs()
    log_path = os.path.join(P['save_path'], 'training_log.txt')
    with open(log_path, 'a', encoding='utf-8') as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f'Training log file: {log_path}')
            print(P, '\n')
            os.environ['CUDA_VISIBLE_DEVICES'] = P['gpu_num']
            print('###### Train start ######')
            try:
                run_train(P)
            except Exception:
                print(traceback.format_exc())
                raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

if __name__ == "__main__":
    main()
