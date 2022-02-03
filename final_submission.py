
import random

def cleaner(PPATH,BPATH,SPATH):
    f = open(PPATH)
    data_f = f.read()
    f.close()
    import pandas as pd
    dfx = pd.read_csv(BPATH,sep='\t',header=None)
    data_lst = data_f.replace(", ",",").split('\n')
    data_lst1 = data_lst
    fleg=False
    data_f = ''
    old_number = 0
    for i in range(len(data_lst1)-1):
        fleg = False
        number, lst_number = data_lst1[i].split(" ")
        if int(number)-old_number>1:
            pet = int(number)-old_number
            for j in range(pet):
                if j!=1:
                    old_number+=1
                    fleg=True
                    data_f+=str(old_number) + " " + str(random.sample([i+1 for i in range(len(dfx.loc[old_number-1][4].split(" ")))],len(dfx.loc[old_number-1][4].split(" ")))).replace(", ",",")+ "\n"
        else:
            data_f+=str(number) +" "+ str(lst_number) + "\n"
        old_number+=1
        if fleg:
            data_f+=str(number) +" "+ str(lst_number) + "\n"
    data_f = data_f.strip()
    f = open(SPATH,"w")
    f.write(data_f)
    f.close()
    print("Done!")


cleaner(".../MIND_dataset/result.txt",".../MINDlarge_test/behaviors.tsv","..../MIND_dataset/prediction.txt")

