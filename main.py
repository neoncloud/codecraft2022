import sys
from frame import parse_init_frame
from task import Scheduler

def read_until_ok():
    input_str = ""
    while True:
        line = input().strip()
        if line == "OK":
            break
        input_str += line+"\n"
    # clear stdin buffer
    sys.stdin.flush()
    return input_str

def print_stdin_and_stderr(s):
    print(s, file=sys.stdout)
    # print(s, file=sys.stderr)

if __name__ == '__main__':
    read_until_ok() # 不需要读地图，第一帧就能读到全部数据
    print("OK")
    init_frame = read_until_ok()
    # with open("test.log", "w") as f:
    #     f.write(init_frame)
    map_obj = parse_init_frame(init_frame)
    scheduler = Scheduler(map_obj)
    init_task = scheduler.init_task()
    print(init_task, file=sys.stderr)
    init_dispatch = scheduler.dispatch()
    print(init_dispatch, file=sys.stderr)
    print_stdin_and_stderr(map_obj.output())
    try:
        while True:
            frame = read_until_ok()
            map_obj.update(frame)
            scheduler.step()
            print_stdin_and_stderr(map_obj.output())
    except EOFError:    # 读到EOF时, 程序结束
        pass