import re
import numpy as np
import argparse

def log_filter(args: argparse.Namespace):
    patterns_to_exclude = [
        "Added request",
        '"POST /v1/completions HTTP/1.1"',
        "Received request cmpl-",
        "[Async] finished",
        "[Async] starting",
        "Tokenize Input Time"
    ]
    before_dot, after_dot = args.filename.split('.', 1)
    
    with open("logs/" + args.filename, "r") as logfile, open("logs/" + before_dot + "_filter." + after_dot, "w") as output_file:
        for line in logfile:
            if not any(pattern in line for pattern in patterns_to_exclude):
                output_file.write(line)

def get_stats(args: argparse.Namespace, pattern):

    with open("logs/" + args.filename, "r") as file:
        # 跳过前10000行
        for _ in range(10000):
            next(file)
        log_data = file.read()

    # 使用正则表达式查找所有匹配的数字
    matches = re.findall(pattern, log_data)
    
    # 将匹配到的数字转换为浮点数
    numbers = [float(match) for match in matches]
    # numbers = [num for num in numbers if num>0]

    # 输出提取到的数字
    # print(numbers)

    if len(numbers) > 0:
        peak = np.max(numbers)
        median = np.median(numbers)
        percent_25 = np.quantile(numbers, 0.25)
        percent_75 = np.quantile(numbers, 0.75)

        print(f"25% Percent: {percent_25:.1f}")
        print(f"50% Percent: {median:.1f}")
        print(f"75% Percent: {percent_75:.1f}")
        print(f"Peak: {peak:.1f}")

        print(f"Iteration num: {len(numbers)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="log")
    args = parser.parse_args()
    
    log_filter(args)

    print("=============KV Cache Usage==============================")
    pattern = r"(\d+\.\d+)% GPU Cache Blocks are used"
    get_stats(args, pattern)
    print("=============Running Deque Number==============================")
    pattern = r"(\d+) Running Requests"
    get_stats(args, pattern)
    print("=============Waiting Deque Number==============================")
    pattern = r"(\d+) Waiting Requests"
    get_stats(args, pattern)
    print("=============Swapped Deque Number==============================")
    pattern = r"(\d+) Swapped Requests"
    get_stats(args, pattern)
    print("=============overall tokens Number==============================")
    pattern = r"(\d+\.\d+) overall tokens"
    get_stats(args, pattern)  
    print("=============prompt tokens Number==============================")
    pattern = r"(\d+) prompt tokens"
    get_stats(args, pattern)    
    print("=============generated tokens Number==============================")
    pattern = r"(\d+\.\d+) generated tokens"
    get_stats(args, pattern)