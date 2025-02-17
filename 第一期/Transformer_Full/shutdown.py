import subprocess
import argparse


def shutdown(time=0):
    # 定义 shell 脚本路径
    script_path = "/mnt/nlp/dh_ai/shell/execution.sh"

    # 调用脚本，并传入参数 --delay 0
    result = subprocess.run(
        [script_path, "--delay", f'{time}'], capture_output=True, text=True)

    # 输出脚本执行的结果
    print(result.stdout)
    print(result.stderr)


if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser(description="解析 --delay 参数的示例脚本")

    # 添加 --delay 参数，并设置默认值为 10
    parser.add_argument('--delay', type=int, default=10,
                        help='指定延迟时间（秒），默认为 2 秒')

    # 解析参数
    args = parser.parse_args()

    shutdown(args.delay)
