import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

score_map = {}

with open('./results/scores_and_preprocessing/final_1024_mean_std_resize_optimized1.csv') as csvfile:
#with open('./results/scores_and_preprocessing/front_resize_optimized.csv') as csvfile:
#with open('./results/scores_and_preprocessing/front_advanced_color.csv') as csvfile:
#with open('./results/scores_and_preprocessing/front_baseline_color.csv') as csvfile:
#with open('./results/scores_and_preprocessing/front_baseline_gray.csv') as csvfile:
#with open('./results/scores_and_preprocessing/front_contrast_strech_color_p2.csv') as csvfile:
#with open('./results/scores_and_preprocessing/front_contrast_strech_color_p10.csv') as csvfile:
#with open('./results/scores_and_preprocessing/front_contrast_strech_gray_p2.csv') as csvfile:
#with open('./results/scores_and_preprocessing/front_equalise_hist_color.csv') as csvfile:       
#with open('./results/scores_and_preprocessing/front_equalise_hist_gray.csv') as csvfile:       
    line_reader = csv.reader(csvfile, delimiter=',')
    for row in line_reader:
        score_map[row[0]] = float(row[1])
        

def calculate_metrics():
    values = list(score_map.values())
        
    score_mean = np.mean(values)
    score_median = np.median(values)
    score_std = np.std(values)
    
    print(" ")
    print("Mean: " + str(score_mean))
    print("Median: " + str(score_median))
    print("Std: " + str(score_std))
    
    n, bins_raw, patches = plt.hist(values, bins=100, facecolor='g')
    plt.xlabel('Dice Scores');
    plt.ylabel('Count')
    plt.title('Initial Histogram of scores running the predictor')
    plt.axis([np.min(values) - 0.005, 1.005, 0, 50])
    plt.grid(True)
    plt.show()

    outlier_list = []
    final_score_list = []
    
    for item in score_map.items():
        value = item[1]
        if not ((value > (score_mean - 2 * score_std)) and (value < (score_mean + 2 * score_std))):
            outlier_list.append(item)
        else:
            final_score_list.append(value)

#    for index in range(len(values)):
#        value = values[index]
#        if not ((value > (score_mean - 2 * score_std)) and (value < (score_mean + 2 * score_std))):
#            outlier_list.append(index)
#        else:
#            final_score_list.append(value)

    score_mean = np.mean(final_score_list)
    score_median = np.median(final_score_list)
    score_std = np.std(final_score_list)

    print("Mean: " + str(score_mean))
    print("Median: " + str(score_median))
    print("Std: " + str(score_std))
    
    print(" ")
    print("Final score list length: " + str(len(final_score_list)))

    plt.hist(final_score_list, bins=bins_raw, facecolor='b')
    plt.xlabel('Dice Scores');
    plt.ylabel('Count')
    plt.title('Final Histogram of scores after removing outliers')
    plt.axis([np.min(values) - 0.005, 1.005, 0, 50])
    plt.grid(True)
    plt.show()

    print(" ")
    print("Removed " + str(len(outlier_list)) + " outliers from the list:")

    for item in outlier_list:
        print(item)
        

if __name__ == '__main__':
    calculate_metrics()
    
#    data = np.load('./input_CP/original_mean.npy')
#    print(data.shape)
#    data = np.squeeze(data, axis=(0))
#    fig = plt.figure(figsize=(32, 32))    
#    imgplt = fig.add_subplot(1, 1, 1)
#    imgplt.set_title("mean")
#    plt.imshow(data)
#    fig.savefig('./mean.jpg')
    
    
    