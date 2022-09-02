import os
import datasets
from datasets import ImageClassification
from dataset.imagenet_labels import IMAGENET2012_CLASSES
from utils import env_vars

PATH = env_vars.get_imagenet_root()
TRAIN_TXT = 'ILSVRC2012_train.txt'
TEST_TXT = 'ILSVRC2012_test.txt'

# row should contains labels, image
class Imagenet1kHf(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    DEFAULT_WRITER_BATCH_SIZE = 1000

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "image": datasets.Value('string'),
                    "label": datasets.ClassLabel(names=list(IMAGENET2012_CLASSES.values())),
                }
            ),
            task_templates=[ImageClassification(image_column="image", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "txt_path": os.path.join(PATH, TRAIN_TXT),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name='val',
                gen_kwargs={
                    "txt_path": os.path.join(PATH, TEST_TXT),
                    "split": "val",
                },
            ),
        ]

    def _generate_examples(self, txt_path, split):
        """Yields examples."""
        # idx = 0
        # for archive in archives:
        #     for path, file in archive:
        #         if path.endswith(".JPEG"):
        #             if split != "test":
        #                 # image filepath format: <IMAGE_FILENAME>_<SYNSET_ID>.JPEG
        #                 root, _ = os.path.splitext(path)
        #                 _, synset_id = os.path.basename(root).rsplit("_", 1)
        #                 label = IMAGENET2012_CLASSES[synset_id]
        #             else:
        #                 label = -1
        #             ex = {"image": {"path": path, "bytes": file.read()}, "label": label}
        #             yield idx, ex
        #             idx += 1
        if split == 'train':
            with open(txt_path, 'r') as ftxt:
                lines = ftxt.readlines()
            for idx, line in enumerate(lines):
                path, class_idx = line.split()
                class_name = os.path.split(os.path.split(path)[0])[-1]
                path = os.path.join(PATH, path)
                class_idx = int(class_idx)
                # with open(path, 'rb') as f:
                #     image_bytes = f.read()
                yield idx, {'image':path, 'label': IMAGENET2012_CLASSES[class_name]}
        elif split == 'val':
            idx = 0
            val_dir = os.path.join(PATH, 'val')
            for path, dirs, files in os.walk(val_dir):
                for filename in files:
                    class_name = os.path.split(path)[-1]
                    if not class_name in IMAGENET2012_CLASSES:
                        raise Exception(class_name, 'is not in classes')
                    file_path = os.path.join(path, filename)
                    yield idx, {'image':file_path, 'label': IMAGENET2012_CLASSES[class_name]}
                    idx += 1
        else:
            raise Exception()

if __name__ == '__main__':
    from datasets import load_dataset
    data = load_dataset('./dataset/imagenet_hf.py', cache_dir='./cache/dataset')
    trainset = data['train']
    testset = data['val']

    import transformers, tqdm
    from dataset.images_hf import ImagesHfDataset, ExamplesToBatchTransform, ViTInputTransform

    extractor = transformers.AutoFeatureExtractor.from_pretrained('facebook/deit-small-distilled-patch16-224')
    dataset = ImagesHfDataset(
        ExamplesToBatchTransform(ViTInputTransform(extractor)),
        ExamplesToBatchTransform(ViTInputTransform(extractor, test=True)),
        batch_size=32,
        name='./dataset/imagenet_hf.py',
        split='train',
        test_split='val',
        num_workers_train=16,
    )

    for batch in tqdm.tqdm(dataset.get_train_iter()):
        print(batch.keys(), batch['pixel_values'][0,0,0,0])