import re
import numpy as np
import argparse

def main(args: argparse.Namespace):
    # 定义正则表达式来匹配百分数的浮点数
    pattern = r"(\d+\.\d+)% GPU Cache Blocks are used"

    # 打开日志文件并读取内容
    with open("logs/" + args.filename, "r") as file:
        # 跳过前4000行
        for _ in range(4000):
            next(file)
        log_data = file.read()

    # 使用正则表达式查找所有匹配的数字
    matches = re.findall(pattern, log_data)

    # 将匹配到的数字转换为浮点数
    numbers = [float(match) for match in matches]
    numbers = [num for num in numbers if num>0]

    # 输出提取到的数字
    print(numbers)

    peak_usage = np.max(numbers)
    median_usage = np.median(numbers)
    usage_25 = np.quantile(numbers, 0.25)
    usage_75 = np.quantile(numbers, 0.75)

    print(f"25% Usage: {usage_25:.1f}")
    print(f"Median Usage: {median_usage:.1f}")
    print(f"75% Usage: {usage_75:.1f}")
    print(f"Peak Usage: {peak_usage:.1f}")

    print(f"Iteration num: {len(numbers)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="log")
    args = parser.parse_args()
    main(args)