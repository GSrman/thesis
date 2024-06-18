from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
import sys


def semantic_similarity(pathA, pathB):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    textA = textB = ""
    with open(pathA, "r") as fileA:
        textA = fileA.read()
    with open(pathB, "r") as fileB:
        textB = fileB.read()

    return 1 - distance.cosine(model.encode([textA])[0],
                               model.encode([textB])[0]
                               )


def main():
    print(semantic_similarity(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
