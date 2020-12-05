import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans

k = 10
calcWords = True
calcTitles = True

clusterTerms = True


titleFile = open("data/science2k-titles.txt")
content = titleFile.read()
titleList = content.split('"\n"')
titleFile.close()
vocabFile = open("data/science2k-vocab.txt")
content = vocabFile.read()
vocabList = content.split('\n')
vocabFile.close()

if clusterTerms:
    wordData = np.load("data/science2k-word-doc.npy")

else:
    wordData = np.load("data/science2k-doc-word.npy")


kmeans = KMeans(n_clusters=k).fit(wordData)

# print(kmeans.cluster_centers_)]

wordFrame = pd.DataFrame(wordData)
wordAvgs = wordFrame.mean(axis=0)
clusterCenters = kmeans.cluster_centers_

def calcDist(pdRow):
    # print(clusterCenters[int(pdRow.loc['label'])])
    return np.linalg.norm(pdRow.iloc[:-1].to_numpy() - clusterCenters[int(pdRow.loc['label'])])


wordFrame['label'] = kmeans.labels_
wordFrame['clusterDist'] = wordFrame.apply(calcDist, axis=1)
wordFrame['clusterDist'] = wordFrame['clusterDist'].astype(float)

for i in range(0, k):
    print()
    print("Cluster " + str(i+1) + ":")
    if calcWords:
        temp = clusterCenters[i] - wordAvgs
        topWords = temp.nlargest(10)
        if clusterTerms:
            print("Top Documents for Cluster " + str(i+1))
            for index in topWords.keys():
                print(titleList[index])
        else:
            print("Top Words for Cluster " + str(i+1))
            for index in topWords.keys():
                print(vocabList[index])

    if calcTitles:
        temp = wordFrame.loc[wordFrame['label'] == i]
        print()
        topWords = temp.nsmallest(10, 'clusterDist')
        if clusterTerms:
            print("Top Words for Cluster " + str(i+1))
            for i in range(0,len(topWords.index)):
                print("-" + vocabList[topWords.index[i]])
            print("This cluster contains " + str(len(temp.index)) + " word(s)")
        else:
            print("Top Titles for Cluster " + str(i+1))
            for i in range(0,len(topWords.index)):
                print("-" + titleList[topWords.index[i]])
            print("This cluster contains " + str(len(temp.index)) + " article(s)")


