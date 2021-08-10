import csv
##nltk.download('punkt')

x = 0
curr_review_text_notlocal = ''
curr_review_text_local = ''

with open('../data/full_balanced.csv', 'r', errors='ignore') as csv_file, open('../data/full_balanced.txt', 'a') as out_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        x = x + 1
##        review_id = row[0]
        user_id = row[0]
        if x == 1:
            curr_user_id = user_id
        review_text = row[1].replace('\r', '').replace('\n', '')
        isLocal = row[2]
 
        if curr_user_id == user_id and isLocal == '0':
            curr_review_text_notlocal = curr_review_text_notlocal + ' ' + review_text

        elif curr_user_id == user_id and isLocal == '1':
            curr_review_text_local = curr_review_text_local + ' ' + review_text

        else:
            if curr_review_text_notlocal:
                output = '%s, %s, %s \n' % (curr_user_id, 0, curr_review_text_notlocal)
                try:
                    out_file.write(output)
                except IOError:
                    print('Error: %s' % output)
            if curr_review_text_local:
                output = '%s, %s, %s \n' % (curr_user_id, 1, curr_review_text_local)
                try:
                    out_file.write(output)
                except IOError:
                    print('Error: %s' % output)

            curr_review_text_notlocal = ''
            curr_review_text_local = ''
            if isLocal == '0':
                curr_review_text_notlocal = review_text
            elif isLocal == '1':
                curr_review_text_local = review_text
            curr_user_id = user_id
                
        if x % 10000 == 0:
            print("row #: %s" % x)
##        if x == 20:
##            break
out_file.close()
csv_file.close()



