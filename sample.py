##
import re
##
re.search('n','\n') #no matches
re.search('n','\\n') # matches with n
re.match('a','abc') # it check for the first character to match
re.findall('abcd','abcdfdsaafc') #finds all character from the string
## Character sets
print(re.search('\w\w','asccvfASD123_').group())  #\w represents [a-zA-Z0-1_] it searches with number of \w 's
print(re.search('\W','.').group()) #\W represents not included in \w

##
#quantifiers
#'+' =1 or more
#

