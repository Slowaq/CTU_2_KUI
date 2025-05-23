import argparse, os
from PIL import Image
import numpy as np
   
K_NEIGHBOURS = 5
NUM_OF_LEVELS = 4
   
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
   
def quantize_levels(vec):
    bucket_size = 256 // NUM_OF_LEVELS
    return np.floor_divide(vec, bucket_size).astype(int)
  
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
    return counts
   
def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    train_dir = os.fsencode(args.train_path)
    out = open(args.o, "w")
   
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
   
    train_vectors = {
        fname: image_flatten(os.path.join(args.train_path, fname))
        for fname in train_labels
    }
  
    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
   
    #num_of_neighb = args.k
    num_of_neighb = K_NEIGHBOURS
    print(f"Running k-NN classifier with k={num_of_neighb}")
   
    final_classifier = {}
    for test in os.listdir(args.test_path):
        if not test.lower().endswith(".png"): continue
        neighbours = []
        test_vector = image_flatten(args.test_path + "/" + test)
   
        for train_fname, train_vec in train_vectors.items():
            dist = distance(train_vec, test_vector)
            label = train_labels[train_fname]      
            neighbours.append((dist, label))
   
        neighbours.sort(key=lambda x: x[0])
        k_nearest = neighbours[:num_of_neighb]
   
        extraction = extract_classes(k_nearest)
        best_label = max(extraction, key=extraction.get)
        final_classifier[test] = best_label
   
        out.write(f"{test}:{best_label}\n")

    out.close()
    return
           
       
if __name__ == "__main__":
    main()