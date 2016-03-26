import os
import pz
import re
    
## retrieve information about the <position> entries
#lst = [] 
#with open('02ac03ae008dd7a247757d69accf253ba7180ccc05bc624067e42cecd892b01b.cfg') as f:
#    for l in f:
#        l = l.rstrip()
#        m = re.search('(@-?\d+) ', l)
#        if m is not None:
#            lst.append(m.group(1))
#        
##print lst
#print 'totoal number of positions: ', len(lst)
#
#unique_lst = list(set(lst))
##print unique_lst
#print 'number of unique positions: ', len(unique_lst)
#
#
## retrieve information about the <name> entries
#lst = [] 
#with open('02ac03ae008dd7a247757d69accf253ba7180ccc05bc624067e42cecd892b01b.cfg') as f:
#    for l in f:
#        l = l.rstrip()
#        m = re.search('@-?\d+ (.*?) ', l)
#        if m is not None:
#            lst.append(m.group(1))
#                  
##print lst
#print 'totoal number of names: ', len(lst)
#
#unique_lst = list(set(lst))
##print unique_lst
#print 'number of unique names: ', len(unique_lst)
#
#
#
#
#G = pz.load('7ab710302c995b3fd72327eb21bcd3f190fdeed81568bb46d4dc2cd9cbf28ea0.fcg.pz')
#
#
#H = pz.load('1.pz')
#H.nodes()
#H.nodes(0)
#len(H.nodes())
#
#
#f = open('bla.txt')
#c = f.read()
#
#with open('bla.txt') as f:
#    i = 0
#    for l in f:
#        print str(i)+':', l.__repr__()
#        i += 1
#        
        

lst = []
file_names = os.listdir('mal')

# file_name = file_names[0]
regexp = '//[ \t]+sha-256:[ \t]+.*\n?(//[ \t]+type:[ \t]+CFG\n*)?'
i = 1
for file_name in file_names[0:1]:    
    f = open(os.path.join('mal', file_name))
    c = f.read()

    m = re.search(regexp, c)
    if m is not None:
        if m.group(0) == c:
            lst.append(i)
        else:
            print str(i)+': pattern not matched!'
    else:
        print str(i)+': pattern not matched!'
            
    i += 1
           
        
# //\s+sha-256:\s+.*\n//\s+type:\s+CFG\n+
c = '// sha-256: 0013cc157f5642d70d95ca0cf08f00521d613640d9b0945876cb45c4af52f865'
c = '// sha-256: 0013cc157f5642d70d95ca0cf08f00521d613640d9b0945876cb45c4af52f865'
m = re.search('aha(\njo)?', c)
m.group(0)
m

m = re.search(regexp, c)
if m is not None:
    if m.group(0) == c:
        print str(i)+': pattern matched!'
else:
    print str(i)+': pattern not matched!'
