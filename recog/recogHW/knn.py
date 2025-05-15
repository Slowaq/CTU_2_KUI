import argparse, os
from PIL import Image
import numpy as np



def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument('-k', type=int, 
                        help='run k-NN classifier (if k is 0 the code may decide about proper K by itself')
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser

def image_flatten(path):
    if path.lower().endswith(".png"):
        return np.array(Image.open(path)).astype(int).flatten()

def distance(im1, im2):
    diff = im1 - im2
    d1 = np.sqrt(np.sum(np.square(diff)))
    d2 = np.linalg.norm(diff)
    return (d1 + d2) / 2

def extract_classes(neighbours):
    labels = [name for (_, name ) in neighbours]
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    # print("COUNTS:", counts)
    return counts

def main():
    parser = setup_arg_parser()
    global args 
    args = parser.parse_args()
    train_dir = os.fsencode(args.train_path)
    print(args.train_path)
    print("Args:", args)

    out = open(args.o, "w")

    train_tuples = {}
    for f in os.listdir(train_dir):
        filename = os.fsdecode(f)
        if filename.endswith(".dsv"):
            with open(str(args.train_path) + "/" + filename, "r") as file:
                for line in file:
                    line = line[:-1]
                    temp = tuple(line.split(":"))
                    train_tuples[temp[0]] = temp[1]
            break

    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    
    print(f"Running k-NN classifier with k={args.k}")
    
    # TODO Train and test the k-NN classifier

    train_vectors = {}
    for fname, label in train_tuples.items():
        path = os.path.join(args.train_path, fname)
        vector  = image_flatten(path)
        train_vectors[fname] = vector

    final_classifier = {}
    for test in os.listdir(args.test_path):
        if not test.lower().endswith(".png"): continue
        neighbours = []
        test_vector = image_flatten(args.test_path + "/" + test)

        for train in os.listdir(args.train_path):
            if not train.lower().endswith(".png"): continue

            # train_vector = image_flatten(args.train_path + "/" + train)
            neighbours.append((distance(train_vectors[train], test_vector), test))

        neighbours = sorted(neighbours, key=lambda x: x [1])[:args.k]
        # print(extract_classes(neighbours))

        extraction = extract_classes(neighbours)
        if len(extraction) == 0:
            print(test)
        best_label = max(extraction, key=extraction.get)
        # print("JHAKO",best_label)
        final_classifier[test] = best_label
        out.write(f"{test}:{train_tuples[best_label]}\n")

    # print(final_classifier)
    out.close()
    return
        
    
if __name__ == "__main__":

    main()
    
