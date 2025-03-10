import os


def get_all_markdown_files(dir_path: str):
    for file in os.listdir(dir_path):
        if file.endswith('.md') and len(file) < 10:
            yield os.path.join(dir_path, file)


def merge_markdown_files(files: list[str], dst_path: str):
    for i, file in enumerate(files):
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(dst_path, 'a', encoding='utf-8') as f:
            if i != 0:
                f.write('\n\n<!-- 文档分割线 -->\n\n')
            f.write(content)


def main():
    dir_path = 'raw_data'
    dst_path = 'merged_data'
    files = get_all_markdown_files(dir_path)
    temp_files = []
    limit = 500
    fragment = 0
    for file in files:
        temp_files.append(file)
        if len(temp_files) == limit:
            merge_markdown_files(temp_files, os.path.join(
                dst_path, f'{fragment}.md'))
            temp_files = []
            fragment += 1

    if len(temp_files) > 0:
        merge_markdown_files(temp_files, os.path.join(
            dst_path, f'{fragment}.md'))


if __name__ == '__main__':
    main()
