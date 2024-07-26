import os
import datasets
import soundfile

class LocalLibriSpeech(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="clean", version=datasets.Version("1.1.0"), description="Cleaned version of LibriSpeech dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="LibriSpeech dataset from local files",
            features=datasets.Features({
                "file": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "text": datasets.Value("string"),
                "speaker_id": datasets.Value("int64"),
                "chapter_id": datasets.Value("int64"),
                "id": datasets.Value("string"),
            }),
            supervised_keys=("file", "text"),
            homepage="http://www.openslr.org/12",
        )

    def _split_generators(self, dl_manager):
        data_dir = os.getenv('LIBRISPEECH_PATH') # 本地数据集路径，需要通过环境变量引入
        return [
            datasets.SplitGenerator(name="train", gen_kwargs={"data_dir": os.path.join(data_dir, "train-clean-100")}),
            datasets.SplitGenerator(name="validation", gen_kwargs={"data_dir": os.path.join(data_dir, "dev-clean")}),
            datasets.SplitGenerator(name="test", gen_kwargs={"data_dir": os.path.join(data_dir, "test-clean")}),
        ]

    def _generate_examples(self, data_dir):
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".trans.txt"):
                    trans_file_path = os.path.join(root, file)
                    with open(trans_file_path, "r") as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            file_id = parts[0]
                            text = " ".join(parts[1:])
                            speaker_id = file_id.split('-')[0]
                            chapter_id = file_id.split('-')[1]
                            file_path = os.path.join(root, file_id + ".txt")
                            audio_data, sample_rate = soundfile.read(os.path.join(root, file_id + ".flac"))
                            audio = {"array": audio_data, "sampling_rate": sample_rate}
                            yield file_id, {
                                "file": file_path,
                                "audio": audio,
                                "text": text,
                                "speaker_id": speaker_id,
                                "chapter_id": chapter_id,
                                "id": file_id,
                            }
