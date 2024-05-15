import json
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree
from path import Path
import cv2

with open('../data/config.json') as f:
    sample_config = json.load(f)

with open('../data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
    # print(word_list)
prefix_tree = PrefixTree(word_list)

for decoder in ['best_path', 'word_beam_search']:
    for img_filename in Path('../data').files('*.png'):
        print("Reading file", img_filename, "with decoder", decoder)

        # read text
        img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
        scale = sample_config[img_filename.basename()]['scale'] if img_filename.basename() in sample_config else 1
        margin = sample_config[img_filename.basename()]['margin'] if img_filename.basename() in sample_config else 0
        read_lines = read_page(img,
                               detector_config=DetectorConfig(scale=scale, margin=margin),
                               line_clustering_config=LineClusteringConfig(min_words_per_line=2),
                               reader_config=ReaderConfig(decoder=decoder, prefix_tree=prefix_tree))
        # print("Image", img)
        # print("Scale", scale)
        # print("Margin", margin)
        # print("ReadLines", read_lines)

        # for read_line in read_lines:
        #     print(' '.join(read_word.text for read_word in read_line))
        # print()
