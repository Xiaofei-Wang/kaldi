# This script is used to analyze the results of finding the mis-transcription techniques
# Copyright: Xiaofei Wang, 2019

import matplotlib.pyplot as plt

text_biased_decode_file = "text_biased_decode"
text_true_file = "text_true"
output_file = "ground_truth.list"
biased_score = "general_score.txt"
refer_file = "reference.txt"
cnt = 0

fp_true = open(text_true_file)
fp_out = open(output_file, 'w')
bias_out = open(biased_score, 'w')
refer_out = open(refer_file, 'w')

# Here, I would build a list to store the file names has transcription errors.
trans_have_error = [line.rstrip('\n') for line in open('mistrans.list')]
trans_corr_rec_corr = [0 for i in range(0, 100, 1)]
trans_wrong_rec_corr = [0 for i in range(0, 100, 1)]
trans_corr_rec_wrong = [0 for i in range(0, 100, 1)]
trans_wrong_rec_wrong = [0 for i in range(0, 100, 1)]
flag = 0

with open(text_biased_decode_file) as fp:
    line = fp.readline()
    cnt = 1
    while line:

        line = line.strip()
        name = line.split()[0]
#        print(name)

        reference = line.replace(name+' ref  ','')
#        print(reference)

        # read the hyp line
        line = fp.readline()
        cnt += 1
        line = line.strip()
        hyp = line.replace(name+' hyp  ','')
#        print(hyp)

        # read the error line
        line = fp.readline()
        line = line.strip()
        cnt += 1

        # read the last line and calculate the WER of each utterance
        line = fp.readline()
        cnt += 1
        line = line.strip()
        nums = line.replace(name+' #csid ','')
        num = [int(i) for i in nums.split()]
        WER = float(sum(num[1:]))/float(sum(num))

        # find the ground truth of utterance with "name"
        line_true = fp_true.readline()
        line_true = line_true.strip()
        truth = line_true.replace(name+' ','')
#        print(truth)
        for i in range(0, 100, 1):
            thres = i/100.0
            if name in trans_have_error and WER < thres:
                trans_wrong_rec_corr[i] += 1
            elif name in trans_have_error and WER >= thres:
                trans_wrong_rec_wrong[i] += 1
            elif name not in trans_have_error and WER >= thres:
                trans_corr_rec_wrong[i] += 1
            else:
                trans_corr_rec_corr[i] += 1

        fp_out.write("{} | {} | {} | {} | {}\n".format(name, hyp, reference, truth, WER))  # python will convert \n to os.linesep
        bias_out.write("{} {}\n".format(name, WER))  # python will convert \n to os.linesep
        if name in trans_have_error:
            flag = 1
        else:
            flag = 0
        refer_out.write("{} {}\n".format(name, flag))  # python will convert \n to os.linesep

        line = fp.readline()
        cnt += 1

print("{} of {} mis-transcribed have not been detected.".format(trans_wrong_rec_corr, trans_wrong_rec_corr[0] + trans_wrong_rec_wrong[0]))
#print("{} of {} mis-transcribed have been detected.".format(trans_wrong_rec_wrong, trans_wrong_rec_corr[0] + trans_wrong_rec_wrong[0]))
print("{} of {} correctly transcribed sentences were wrongly recognize. False Alarm!".format(trans_corr_rec_wrong, trans_corr_rec_wrong[0] + trans_corr_rec_corr[0]))
#print("{} of {} correctly transcribed sentences were correctly recognize. No Worry! ".format(trans_corr_rec_corr, trans_corr_rec_wrong[0] + trans_corr_rec_corr[0]))

trans_wrong_rec_corr = [i/1810.0 for i in trans_wrong_rec_corr]
trans_corr_rec_wrong = [i/10093.0 for i in trans_corr_rec_wrong]
print("{} of {} mis-transcribed have not been detected.".format(trans_wrong_rec_corr, trans_wrong_rec_corr[0] + trans_wrong_rec_wrong[0]))
print("{} of {} correctly transcribed sentences were wrongly recognize. False Alarm!".format(trans_corr_rec_wrong, trans_corr_rec_wrong[0] + trans_corr_rec_corr[0]))

fp.close()
fp_true.close()
fp_out.close()
bias_out.close()
refer_out.close()

trans_corr_rec_corr_kl = [0 for i in range(0, 100, 1)]
trans_wrong_rec_corr_kl = [0 for i in range(0, 100, 1)]
trans_corr_rec_wrong_kl = [0 for i in range(0, 100, 1)]
trans_wrong_rec_wrong_kl = [0 for i in range(0, 100, 1)]

KL_div_file = 'KL_sys_medfilt_15_phone_window_0.txt'
with open(KL_div_file) as fp:
    line = fp.readline()
    while line:
        line = line.strip()
        name = line.split()[0]
        value = float(line.split()[1])
        for i in range(0, 100, 1):
            thres_kl = i/5.0
            if name in trans_have_error and value < thres_kl:
                trans_wrong_rec_corr_kl[i] += 1
            elif name in trans_have_error and value >= thres_kl:
                trans_wrong_rec_wrong_kl[i] += 1
            elif name not in trans_have_error and value >= thres_kl:
                trans_corr_rec_wrong_kl[i] += 1
            else:
                trans_corr_rec_corr_kl[i] += 1

        line = fp.readline()

print("{} of {} mis-transcribed have not been detected.".format(trans_wrong_rec_corr_kl, trans_wrong_rec_corr_kl[0] + trans_wrong_rec_wrong_kl[0]))
#print("{} of {} mis-transcribed have been detected.".format(trans_wrong_rec_wrong_kl, trans_wrong_rec_corr_kl[0] + trans_wrong_rec_wrong_kl[0]))
print("{} of {} correctly transcribed sentences were wrongly recognize. False Alarm!".format(trans_corr_rec_wrong_kl, trans_corr_rec_wrong_kl[0] + trans_corr_rec_corr_kl[0]))
#print("{} of {} correctly transcribed sentences were correctly recognize. No Worry! ".format(trans_corr_rec_corr_kl, trans_corr_rec_wrong_kl[0] + trans_corr_rec_corr_kl[0]))

trans_wrong_rec_corr_kl = [i/1810.0 for i in trans_wrong_rec_corr_kl]
trans_corr_rec_wrong_kl = [i/10093.0 for i in trans_corr_rec_wrong_kl]
print("{} of {} mis-transcribed have not been detected.".format(trans_wrong_rec_corr_kl, trans_wrong_rec_corr_kl[0] + trans_wrong_rec_wrong_kl[0]))
print("{} of {} correctly transcribed sentences were wrongly recognize. False Alarm!".format(trans_corr_rec_wrong_kl, trans_corr_rec_wrong_kl[0] + trans_corr_rec_corr_kl[0]))


#tmp = trans_wrong_rec_corr + trans_wrong_rec_wrong
#trans_wrong_rec_corr = [x/y for x, y in zip(trans_wrong_rec_corr, tmp)]
#trans_wrong_rec_wrong = [x/y for x, y in zip(trans_wrong_rec_wrong, tmp)]

#plt.plot(trans_wrong_rec_wrong, trans_corr_rec_corr)
#plt.show()