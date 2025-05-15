import argparse
import os, math
import sys
import numpy as np
from PIL import Image

NUM_OF_LEVELS = 4
LAPLACE_CORRECTION = 2
TRAINING_RATIO = 0.7
      
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser

def apriori_probab(file):
    class_counts = {}
    for line in file:
        line = line[:-1]
        temp = tuple(line.split(":"))
        class_counts[temp[1]] = class_counts.get(temp[1], 0) + 1
    
    total = sum(class_counts.values())
    class_probab = {}
    for label, cnt in class_counts.items():
        probab = class_counts[label] / total
        log_probab = math.log(probab)
        class_probab[label] = (probab, log_probab)

    return (class_counts, class_probab)

def quantize_levels(vec):
    bucket_size = 256 // NUM_OF_LEVELS
    return np.floor_divide(vec, bucket_size).astype(int)

def image_flatten(path):
    if path.lower().endswith(".png"):
        return np.array(Image.open(path)).astype(int).flatten()

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    train_dir = os.fsencode(args.train_path)
    out = open(args.o, "w")
    resolution = int(args.train_path.split("_")[-1])

    truth_path = os.path.join(args.train_path, "truth.dsv")
    with open(truth_path) as f:
        class_counts, apriori = apriori_probab(f)

    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print("Image resolution:", resolution, "x", resolution)
    print('Output file:', args.o)
    print("Running Naive Bayes classifier\n")

    train_labels = {}
    for f in os.listdir(train_dir):
        filename = os.fsdecode(f)
        if filename.endswith(".dsv"):
            with open(str(args.train_path) + "/" + filename, "r") as file:
                for line in file:
                    line = line[:-1]
                    temp = tuple(line.split(":"))
                    train_labels[temp[0]] = temp[1]
            break
    import random

    # split data into training and validation set
    all_files = list(train_labels.keys())
    random.seed(42)       
    random.shuffle(all_files)

    split_idx = int(TRAINING_RATIO * len(all_files))
    train_files = all_files[:split_idx]
    val_files   = all_files[split_idx:]

    print(f"Using {len(train_files)} files for training and {len(val_files)} for validation!")

    train_vectors = {}
    for fname, _ in train_labels.items():
        path = os.path.join(args.train_path, fname)
        vector  = image_flatten(path)
        train_vectors[fname] = vector
    
    quantized_train_vec = {}
    for fname, vec in train_vectors.items():
        quantized_train_vec[fname] = quantize_levels(vec)

    # Training
    classes = sorted(set(train_labels.values()))
    num_pixels = resolution * resolution
    counts = {
        s: np.zeros((num_pixels, NUM_OF_LEVELS), dtype=int)
        for s in classes
    }

    train_counts = { s: 0 for s in classes }
    for fname in train_files:
        train_counts[ train_labels[fname] ] += 1

    for fname in train_files:
        label = train_labels[fname]
        vec = quantized_train_vec[fname]  
        for idx, qi in enumerate(vec):
            counts[label][idx, qi] += 1

    P_xi_given_s = {}
    logP_xi_given_s = {}

    for s in classes:
        P = np.zeros((num_pixels, NUM_OF_LEVELS), dtype=float)
        logP = np.zeros((num_pixels, NUM_OF_LEVELS), dtype=float)
        denom = train_counts[s] + LAPLACE_CORRECTION * NUM_OF_LEVELS
        for i in range(num_pixels):
            for q in range(NUM_OF_LEVELS):
                numer = counts[s][i, q] + LAPLACE_CORRECTION
                P[i, q] = numer / denom
                logP[i, q] = math.log(numer) - math.log(denom)
        P_xi_given_s[s] = P
        logP_xi_given_s[s] = logP

    # Validation
    if TRAINING_RATIO != 1:
        correct = 0
        for fname in val_files:
            vec = quantize_levels(image_flatten(os.path.join(args.train_path, fname)))
            score = { s: apriori[s][1] for s in classes }
            for qi in vec:
                for s in classes:
                    score[s] += logP_xi_given_s[s][i, qi]
            pred = max(score, key=score.get)
            if pred == train_labels[fname]: correct += 1

        val_acc = correct / len(val_files)
        print(f"Validation accuracy: {val_acc:.2%}")

    for test in os.listdir(args.test_path):
        if not test.lower().endswith(".png"): continue
        vec = quantize_levels(image_flatten(args.test_path + "/" + test))  

        score = { s: apriori[s][1] for s in classes}

        for i, qi in enumerate(vec):
            for s in classes:
                score[s] += logP_xi_given_s[s][i, qi]

        pred = max(score, key=score.get)
        out.write(f"{test}:{pred}\n")
    
if __name__ == "__main__":
    main()
