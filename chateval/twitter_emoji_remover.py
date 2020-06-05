file_name = "./twitter.txt"

file = open(file_name, 'r', encoding='utf-8')
file_out = open("twitter_pre.txt", 'w')

for line in file:
    line = str(line).replace('\n', '')
    cleaned_data = str(line).encode('ascii', 'ignore').decode('utf8')
    file_out.write("%s\n" % (cleaned_data))
