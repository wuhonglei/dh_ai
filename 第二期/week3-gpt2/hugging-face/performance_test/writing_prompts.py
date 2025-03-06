import datasets
import os

# 打印当前路径
print('os.getcwd()', os.getcwd())


class WritingPromptsConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class WritingPrompts(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "prompt": datasets.Value("string"),
                "story": datasets.Value("string"),
            })
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "source_file": "train.wp_source",
                    "target_file": "train.wp_target",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "source_file": "test.wp_source",
                    "target_file": "test.wp_target",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "source_file": "valid.wp_source",
                    "target_file": "valid.wp_target",
                }
            ),
        ]

    def _generate_examples(self, source_file, target_file):
        with open(source_file, encoding="utf-8") as f_source, \
                open(target_file, encoding="utf-8") as f_target:
            for idx, (prompt, story) in enumerate(zip(f_source, f_target)):
                yield idx, {
                    "prompt": prompt.strip(),
                    "story": story.strip(),
                }
