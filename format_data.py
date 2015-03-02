sz = len('00000000000000000011000000000000');
def st_wrt_to_js(ln):
    f = open('fulldata.js','w')
    f.write(ln)
    f.close()

def wrt_to_js(ln):
    f = open('fulldata.js','a')
    f.write(ln)
    f.close()
    
st_wrt_to_js('var raw = {\'rowsz\':'+str(sz)+', \'0x\' :\"')
path = 'data.txt'
ct = 0
max = 10**6;
with open(path, 'r') as f:
    for line in f:
        if len(line)== 3:
            rt = '\",\''+str(ct)+'y\' :'+line[1]
            wrt_to_js(rt)
            ct = ct+1
            if ct > max:
                break
            rt = ',\''+str(ct)+'x\' :\" '
            wrt_to_js(rt)
        elif len(line)== 33:
            line = line[0:-1]
            wrt_to_js(line)
wrt_to_js('};')