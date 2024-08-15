import subprocess
import sys
# print(sys.executable)
import time
from multiprocessing import Pool
from pathlib import Path

import pytest
import requests


def _query_server(prompt: str, max_tokens: int = 5) -> dict:
    response = requests.post("http://localhost:8000/v1/completions",
                             json={
                                 "model": "/root/autodl-tmp/model_dir/opt-125m",
                                 "prompt": prompt,
                                 "max_tokens": max_tokens,
                                 "temperature": 0,
                                 "ignore_eos": True
                             })
    response.raise_for_status()
    return response.json()


def _query_server_long(prompt: str) -> dict:
    return _query_server(prompt, max_tokens=500)


# 启动server
# 在test_api_server之前执行
@pytest.fixture  # 设置测试环境（用于准备测试数据、初始化状态、创建测试依赖等）
def api_server(tokenizer_pool_size: int, engine_use_ray: bool,
               worker_use_ray: bool):

    # 要执行的python脚本路径
    # script_path = Path(__file__).parent.joinpath("api_server_async_engine.py").absolute()
    commands = [
        sys.executable, "-u", "/root/autodl-tmp/vllm/vllm/entrypoints/openai/api_server.py", 
        "--model", "/root/autodl-tmp/model_dir/opt-125m", 
        "--tokenizer-pool-size", str(tokenizer_pool_size)
    ]
    if engine_use_ray:
        commands.append("--engine-use-ray")
    if worker_use_ray:
        commands.append("--worker-use-ray")
    # 启动一个新进程，执行commands的命令
    uvicorn_process = subprocess.Popen(commands)
    yield
    uvicorn_process.terminate()


# 使用下面的参数组合, 多次调用test_api_server()
@pytest.mark.parametrize("tokenizer_pool_size", [0, 2])
@pytest.mark.parametrize("worker_use_ray", [False])
@pytest.mark.parametrize("engine_use_ray", [False])
def test_api_server(api_server, tokenizer_pool_size: int, worker_use_ray: bool,
                    engine_use_ray: bool):
    """
    Run the API server and test it.

    We run both the server and requests in separate processes.

    We test that the server can handle incoming requests, including
    multiple requests at the same time, and that it can handle requests
    being cancelled without crashing.
    """
    print("Start: test_api_server")
    with Pool(32) as pool:
        # Wait until the server is ready
        prompts = ["warm up"] * 1
        result = None
        while not result:
            try:
                for r in pool.map(_query_server, prompts):
                    result = r
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        # Actual tests start here
        # Try with 1 prompt
        for result in pool.map(_query_server, prompts):
            assert result

        # num_aborted_requests = requests.get(
        #     "http://localhost:8000/stats").json()["num_aborted_requests"]
        # assert num_aborted_requests == 0

        # Try with 100 prompts
        prompts = ["test prompt"] * 100
        for result in pool.map(_query_server, prompts):
            assert result

    with Pool(32) as pool:
        # Cancel requests
        prompts = ["canceled requests"] * 100
        pool.map_async(_query_server_long, prompts)
        time.sleep(0.01)
        pool.terminate()
        pool.join()

        # check cancellation stats
        # give it some times to update the stats
        time.sleep(1)

        # num_aborted_requests = requests.get(
        #     "http://localhost:8000/stats").json()["num_aborted_requests"]
        # assert num_aborted_requests > 0

    # check that server still runs after cancellations
    with Pool(32) as pool:
        # Try with 100 prompts
        prompts = ["test prompt after canceled"] * 100
        for result in pool.map(_query_server, prompts):
            assert result
