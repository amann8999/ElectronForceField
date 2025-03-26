import bs4 as bs
import urllib.request
import pandas as pd
import final_data

total = 0
good = 0
mols = set()
for num in range(37):
    url='https://cccbdb.nist.gov/listbyatom2x.asp?atno='+str(num)
    dfs = pd.read_html(url)
    mylist = dfs[1]
    for mol in mylist['Molecule']:
        print(mol)
        if mol not in mols:
            mols.add(mol)
            total += 1
            if final_data.save_data(str(mol), 'out.txt', 1):
                good += 1
            print(str(good), '/', str(total))
    for mol in mylist['Cation']:
        print(mol)
        if mol not in mols:
            mols.add(mol)
            total += 1
            if final_data.save_data(str(mol), 'out.txt', 1):
                good += 1
            print(str(good), '/', str(total))
    for mol in mylist['Anion']:
        print(mol)
        if mol not in mols:
            mols.add(mol)
            total += 1
            if final_data.save_data(str(mol), 'out.txt', 1):
                good += 1
            print(str(good), '/', str(total))
