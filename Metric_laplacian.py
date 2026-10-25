# Apply patch for grakel compatibility with newer numpy
import patch_grakel

import os
import Function

seed=Function.big_dataset_seed_top()
filecom = ["dataset/amazon-1.90.cmty.txt","dataset/dblp-1.90.cmty.txt","dataset/facebook-1.90.cmty.txt","dataset/twitter-1.90.cmty.txt"]
filenedge = ['dataset/amazon-1.90.ungraph.txt', 'dataset/dblp-1.90.ungraph.txt', 'dataset/facebook-1.90.ungraph.txt',
         'dataset/twitter-1.90.ungraph.txt']
path=[["SLSS_laplacian/amazon/1/","SLSS_laplacian/amazon/2/","SLSS_laplacian/amazon/3/","SLSS_laplacian/amazon/4/","SLSS_laplacian/amazon/5/"]
    ,["SLSS_laplacian/dblp/1/","SLSS_laplacian/dblp/2/","SLSS_laplacian/dblp/3/","SLSS_laplacian/dblp/4/","SLSS_laplacian/dblp/5/"]
    ,["SLSS_laplacian/facebook/1/","SLSS_laplacian/facebook/2/","SLSS_laplacian/facebook/3/","SLSS_laplacian/facebook/4/","SLSS_laplacian/facebook/5/"]
    ,["SLSS_laplacian/twitter/1/","SLSS_laplacian/twitter/2/","SLSS_laplacian/twitter/3/","SLSS_laplacian/twitter/4/","SLSS_laplacian/twitter/5/"]
    ]
dataname=['amazon','dblp','facebook','twitter',"lj"]
datasets=list(zip(filecom,filenedge,path,dataname))

for dataset in datasets:
    print("Dataset:",dataset[3])
    for data in dataset[2]:
        print(data)
        know=[]
        node_list1, node, com_index, node_com_index = Function.com_node_index_list_list(dataset[0], dataset[1])
        for root, dirs, files in os.walk(data):
            for filename in files:
                file = open(data + filename, "r")
                lines = file.readlines()
                a1 = []
                for line in lines:
                    lines1 = line.strip("[]\n").split(',')
                    if len(lines1) == 1:
                        lines1 = []
                    else:
                        lines1 = [int(a) for a in lines1]
                    know.append(lines1)
        ff = []
        jj = []
        rr = []
        pp = []
        for i in know:
            p1 = []
            f1 = []
            r1 = []
            j1 = []
            try:
                a = node_com_index[i[0]]
            except:
                pass
            b = []
            for jjjj in a:
                b.append(com_index[jjjj])
            for jjj in b:
                r, p, f, j = Function.compare_comm(i, jjj)
                f1.append(f)
                r1.append(r)
                p1.append(p)
                j1.append(j)
            ff.append(round(sum(f1) / len(f1), 6))
            rr.append(round(sum(r1) / len(r1), 6))
            pp.append(round(sum(p1) / len(p1), 6))
            jj.append(round(sum(j1) / len(j1), 6))
        print(len(rr))
        print("avg_r:", round(sum(rr) / len(rr), 6))
        print("avg_p:", round(sum(pp) / len(pp), 6))
        print("avg_f:", round(sum(ff) / len(ff), 6))
        print("avg_j:", round(sum(jj) / len(jj), 6))
        # print(round(sum(rr) / len(rr), 4))
        # print(round(sum(pp) / len(pp), 4))
        # print(round(sum(ff) / len(ff), 4))
        # print(round(sum(jj) / len(jj), 4))