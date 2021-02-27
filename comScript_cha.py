import subprocess
import psutil
import time
import argparse


def isRuningPid(pid):
    try:
        s = psutil.Process(pid)
        return True
    except psutil.NoSuchProcess:
        return False


def parse():
    parser = argparse.ArgumentParser(description='arguments of script')
    parser.add_argument('--runningPid', default=0, type=int, metavar='P',
                        help='run the program after the pid is over')

    args = parser.parse_args()

    return args

args = parse()
pid = args.runningPid

def main():

    linux_commod = ['python3 -u Train_FrameAttention.py --lr 0.00001 --weight-decay 0.1 --accumulation_step 1 --epochs 50 --non_local_pos 0 2>&1 | tee ale-4w0.1.log',
                    'python3 -u Train_FrameAttention.py --lr 0.00001 --weight-decay 0.1 --accumulation_step 1 --epochs 50 --non_local_pos 4 2>&1 | tee ale-4w0.01.log',
                    'python3 -u Train_FrameAttention.py --lr 0.00001 --weight-decay 0.1 --accumulation_step 1 --epochs 50 --non_local_pos 3 2>&1 | tee ale-4w0.001.log',
                    'python3 -u Train_FrameAttention.py --lr 0.00001 --weight-decay 0.1 --accumulation_step 1 --epochs 50 --non_local_pos 2 2>&1 | tee ale-4w0.0001.log',
                    'python3 -u Train_FrameAttention.py --lr 0.00001 --weight-decay 0.1 --accumulation_step 1 --epochs 50 --non_local_pos 1 2>&1 | tee ale-4w0.0001.log',

                   # 'python3 -u Train_FrameAttention.py --lr 0.0001 --weight-decay 0.1 --accumulation_step 1 --data_time 1 2>&1 | tee a0.1_lre-3.log',
                   # 'python3 -u Train_FrameAttention.py --lr 0.0001 --weight-decay 0.01 --accumulation_step 1 --data_time 1 2>&1 | tee a0.01_lre-3.log',
                   # 'python3 -u Train_FrameAttention.py --lr 0.0001 --weight-decay 0.001 --accumulation_step 1 --data_time 1 2>&1 | tee a0.001_lre-3.log',
                   # 'python3 -u Train_FrameAttention.py --lr 0.0001 --weight-decay 0.0001 --accumulation_step 1 --data_time 1 2>&1 | tee a0.0001_lre-3.log',
                    ]

    while True:
        time.sleep(1)
        if isRuningPid(pid) == False:
        
            for i_commod in linux_commod:
                print(i_commod)
                result = subprocess.getstatusoutput(i_commod)
                print(result[1])
                
            break
        

if __name__ == "__main__":
    main()