import os


def get_all_markdown_files(dir_path: str) -> list[str]:
    files = []
    for file in os.listdir(dir_path):
        if file.endswith('.md'):
            files.append(os.path.join(dir_path, file))
    return files


def empty_file(file_path: str):
    if not os.path.exists(file_path):
        return

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('')


def merge_markdown_files(files: list[str], dst_path: str):
    for i, file in enumerate(files):
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            # temp_content = content.lower().replace('剑指offer', '')
            # if 'offer' not in temp_content:
            #     continue

        if i == 0:
            empty_file(dst_path)

        with open(dst_path, 'a', encoding='utf-8') as f:
            if i != 0:
                f.write('\n\n<!-- 文档分割线 -->\n\n')
            f.write(content)


def main(name: str):
    dir_path = 'boss直聘'
    dst_path = 'merged_data'
    files = get_all_markdown_files(dir_path)
    temp_files = []
    limit = 500
    fragment = 0
    for file in files:
        temp_files.append(file)
        if len(temp_files) == limit:
            merge_markdown_files(temp_files, os.path.join(
                dst_path, f'{name}_{fragment}.md'))
            temp_files = []
            fragment += 1

    if len(temp_files) > 0:
        merge_markdown_files(temp_files, os.path.join(
            dst_path, f'{name}_{fragment}.md'))


if __name__ == '__main__':
    # main('nowcoder')
    main('boss直聘')
