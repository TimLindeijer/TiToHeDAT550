# Task 1: Check-Worthiness in Multimodal and Unimodal Content

The aim of this task is to determine whether a claim in a tweet is worth fact-checking. Typical approaches to make that decision require to either resort to the judgments of professional fact-checkers or to human annotators to answer several auxiliary questions such as "does it contain a verifiable factual claim?", and "is it harmful?", before deciding on the final check-worthiness label.

This year we offer two kinds of data, which translate to two subtasks:
- **Subtask 1A (Multimodal):** The tweets to be judged include both a text snippet and an image.
- **Subtask 1B (Multigenre (Unimodal) - Text):** The tweets and/or transcriptions are to be judged based only on text.

Subtask 1A is offered in Arabic and English, whereas Subtask 1B is offered in Arabic, English and Spanish.


__Table of contents:__
<!-- - [Evaluation Results](#evaluation-results) -->
- [Submission Guidelines](#submission-guidelines)
- [List of Versions](#list-of-versions)
- [Contents of the Directory](#contents-of-the-directory)
- [File Format](#file-format)
	- [Subtask 1A (Multimodal (text + image) -- Tweets): Check-Worthiness of multimodal content](#subtask-1a-multimodal-text-image-tweets-check-worthiness-of-multimodal-content)
		- [Input Data Format](#input-data-format)
	- [Subtask 1B (Multigenre (Unimodal) - Text): Check-Worthiness of multigenre unimodal content](#subtask-1b-multigenre-unimodal-text-check-worthiness-of-multigenre-unimodal-content)
		- [Input Data Format - Tweets](#input-data-format-unimodal-text-tweets)
		- [Input Data Format - Political Debates](#input-data-format-unimodal-text-political-debates)
	- [Output Data Format](#output-data-format)
- [Format Checkers](#format-checkers)
- [Scorers](#scorers)<!-- - [Evaluation Metrics](#evaluation-metrics) -->
- [Baselines](#baselines)<!-- - [Submission guidelines](#submission-guidelines) -->
- [Credits](#credits)



<!-- ## Evaluation Results
Submitted results will be available after the system submission deadline.
Kindly find the leaderboard released in this google sheet, [link](http://shorturl.at/nuCOS). you can find in the tab labeled "Task 1".
 -->
## Submission Guidelines:
- Make sure that you create one account for each team, and submit runs through one account only.
- The last file submitted to the leaderboard will be considered as the final submission.
- Name of the output file has to be `subtask1[A/B]_lang.tsv` with `.tsv` extension (e.g., subtask1B_arabic.tsv); otherwise, you will get an error on the leaderboard. Subtasks are 1A, 1B. For subtask 1A, there are two languages (Arabic, and English). For subtask 1B, there are three languages (Arabic, Spanish, and English).
- You have to zip the tsv, `zip subtask1B_arabic.zip subtask1B_arabic.tsv` and submit it through the codalab page.
It is required to submit the team name and method description for each submission. **Your team name here must EXACTLY match that used during CLEF registration.**
- You are allowed to submit max 200 submissions per day for each subtask.
- We will keep the leaderboard private till the end of the submission period, hence, results will not be available upon submission. All results will be available after the evaluation period.

**Please submit your results on test data here: https://codalab.lisn.upsaclay.fr/competitions/12936**


## List of Versions
* __[2023/05/04]__
  - Training/Dev/Dev_Test data update for subtasks 1A English
* __[2023/05/03]__
  - Released test set.

* __subtask-1B [2023/02/08]__
  - Training/Dev/Dev_Test data for subtasks 1B released for English.

* __subtask-1A [2022/12/28]__
  - Training/Dev/Dev_Test data for subtasks 1A released for English.

* __subtask-1A [2022/12/20]__
  - Training/Dev/Dev_Test data for subtask 1A released for Arabic.

* __subtask-1B [2022/11/23]__
  - Training/Dev/Dev_Test data for subtask 1B released for Arabic and Spanish.


## Contents of the Directory
<!-- In each directory, we provide subtask-specific zip files. Each zip file contains train, dev, and dev_test data released with the tweets and the labels assigned. We provide a single JSON file for the majority of the languages. The tweet id can be used to match the data between the TSV file and the JSON files.

**Notice** Many instances in the TSV file might not have a corresponding entry in the JSON file. This is due to the deletion of tweets during the compilation of the datasets.
 -->
* Main folder: [data](./data)
  	This directory contains files for all languages and subtasks.

* Main folder: [baselines](./baselines)<br/>
	Contains scripts provided for baseline models of the tasks.
* Main folder: [format_checker](./format_checker)<br/>
	Contains scripts provided to check format of the submission file.
* Main folder: [scorer](./scorer)<br/>
	Contains scripts provided to score output of the model when provided with label (i.e., dev and dev_test sets).

* [README.md](./README.md) <br/>
	This file!


## File Format

### Subtask 1A (Multimodal (text + image) -- Tweets): Check-Worthiness of multimodal content

#### Input Data Format

For English, we have three splits train, dev and dev_test jsonl files. Each file has lines of dictionary objects containing the tweets, labels and other information. The text encoding is UTF-8. Each dictionary object has the following keys:

tweet_id, tweet_url, tweet_text, ocr_text, class_label, image_path, image_url

All the images are segregated under the folder `images_labeled` according to the split.

The unlabeled data is available at [Link](https://drive.google.com/drive/folders/1C-yh-VGHY8wZiW0tTVXdeljGyBPuEt03?usp=sharing).

**Examples:**
> {"tweet_id": "1241689641486442496", "tweet_url": "https://twitter.com/user/status/1241689641486442496", "tweet_text": "Our team in Elliot Lake is deployed full force including at No Frills. We are onsite to provide crowd control and limit the number of people in the store at any given time.\n\nPlease respect the guards and their instructions as we get through all of this together. https://t.co/9rk7gEW0Mb", "ocr_text": "PHARMACY\nWON'T BE BEAT\ndde ntw\nFHINRR\n", "class_label": "No", "image_path": "images_labeled/train/1241689641486442496.jpg", "image_url": "http://pbs.twimg.com/media/ETteJOlU4AA_WrU.jpg"}<br/>
> {"tweet_id": "1033455270813421571", "tweet_url": "https://twitter.com/user/status/1033455270813421571", "tweet_text": "What is the purpose of the EU Emissions Trading System?\nhttps://t.co/nb9jeqOkzU\n#climatechange #climateaction \n#environment \n#energy https://t.co/Z9BfYnxEcw", "ocr_text": "Global Climate Change Think & Act Tank 2017. GCCThinkActTank\nPhoto: Anne-Maria\nYritys Https://www.leadingwithpassion.org\n\"EC. Climate Change Key Terms. EU Emissions Trading Scheme (EU ETS) = Company-level “cap-\nand-trade\" system of allowances for emitting carbon dioxide and other greenhouse gases,\nlaunched by the EU in 2005.\"\n", "class_label": "No", "image_path": "images_labeled/train/1033455270813421571.jpg", "image_url": "http://pbs.twimg.com/media/DleSF3wXoAEaHED.jpg"}<br/>
> {"tweet_id": "1204873106294214657", "tweet_url": "https://twitter.com/user/status/1204873106294214657", "tweet_text": "5 Charged in New Jersey in $722 Million Cryptocurrency Ponzi Scheme by Michael Levenson https://t.co/50RMU2bDC2 https://t.co/29yqanGTm0", "ocr_text": "", "class_label": "Yes", "image_path": "images_labeled/train/1204873106294214657.jpg", "image_url": "http://pbs.twimg.com/media/ELiRtGzWsAAdY8s.jpg"}
>
> ... <br/>

<!-- #### Output Data Format -->


### Subtask 1B (Multigenre (Unimodal) - Text): Check-Worthiness of multigenre unimodal content


#### Input Data Format (Unimodal - Text -- Tweets)
For **Arabic**, and **Spanish** we use the same data format in the train, dev and dev_test files. Each file is TAB seperated (TSV file) containing the tweets and their labels. The text encoding is UTF-8. Each row in the file has the following format:

> tweet_id <TAB> tweet_url <TAB> tweet_text <TAB> class_label

Where: <br>
* tweet_id: Tweet ID for a given tweet given by Twitter <br/>
* tweet_url: URL to the given tweet <br/>
* tweet_text: content of the tweet <br/>
* class_label: *Yes* and *No*


**Examples:**
> 1235648554338791427	https://twitter.com/A6Asap/status/1235648554338791427	COVID-19 health advice⚠️ https://t.co/XsSAo52Smu	No<br/>
> 1235287380292235264	https://twitter.com/ItsCeliaAu/status/1235287380292235264	There's not a single confirmed case of an Asian infected in NYC. Stop discriminating cause the virus definitely doesn't. #racist #coronavirus https://t.co/Wt1NPOuQdy	Yes<br/>
> 1236020820947931136	https://twitter.com/ddale8/status/1236020820947931136	Epidemiologist Marc Lipsitch, director of Harvard's Center for Communicable Disease Dynamics: “In the US it is the opposite of contained.' https://t.co/IPAPagz4Vs	Yes<br/>
> ... <br/>

Note that the gold labels for the task are the ones in the *class_label* column.



#### Input Data Format (Unimodal - Text -- Political debates)
For **English** we use the same data format in the train, dev and dev_test files. Each file is TAB seperated (TSV file) containing the tweets and their labels. The text encoding is UTF-8. Each row in the file has the following format:

> sentence_id <TAB> text <TAB> class_label

Where: <br>
* sentence_id: sentence id for a given political debate <br/>
* text: sentence's text <br/>
* class_label: *Yes* and *No*


**Examples:**
> 30313	And so I know that this campaign has caused some questioning and worries on the part of many leaders across the globe.	No<br/>
> 19099	"Now, let's balance the budget and protect Medicare, Medicaid, education and the environment."	No<br/>
> 33964	I'd like to mention one thing.	No<br/>
> ... <br/>

Note that the gold labels for the task are the ones in the *class_label* column.


### Output Data Format
For both subtasks **1A**, and **1B** and for all languages (**Arabic**, **English**, and **Spanish**) the submission files format is the same.

The expected results file is a list of tweets/transcriptions with the predicted class label.

The file header should strictly be as follows:

> **id <TAB> class_label <TAB> run_id**

Each row contains three TAB separated fields:

> tweet_id or id <TAB> class_label <TAB> run_id

Where: <br>
* tweet_id or id: Tweet ID or sentence id for a given tweet given by Twitter or coming from political debates given in the test dataset file. <br/>
* class_label: Predicted class label for the tweet. <br/>
* run_id: String identifier used by participants. <br/>

Example:
> 1235648554338791427	No  Model_1<br/>
> 1235287380292235264	Yes  Model_1<br/>
> 1236020820947931136	No  Model_1<br/>
> 30313	No  Model_1<br/>
> ... <br/>


## Format Checkers

### Subtask 1A and 1B
The checker for the task is located in the [format_checker](./format_checker) module of the project.
To launch the checker script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

The format checker verifies that your generated results files complies with the expected format.
To launch it run:

> python3 format_checker/subtask_1.py --pred-files-path <path_to_result_file_1 path_to_result_file_2 ... path_to_result_file_n> <br/>

`--pred-files-path` is to be followed by a single string that contains a space separated list of one or more file paths.

__<path_to_result_file_n>__ is the path to the corresponding file with participants' predictions, which must follow the format, described in the [Output Data Format](#output-data-format) section.

Note that the checker can not verify whether the prediction files you submit contain all tweets, because it does not have access to the corresponding gold file.


## Scorers

The scorer for the task is located in the [scorer](./scorer) module of the project.
To launch the script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

Launch the scorer for the subtask as follows:
> python3 scorer/subtask_1.py --gold-file-path=<path_gold_file> --pred-file-path=<predictions_file> --subtask=<name_of_the_subtask> --lang=<language><br/>

`--subtask` expects one of two options **A** or **B** to indicate the subtask for which to score the predictions file.

`--lang` expects one of three options **arabic** or **english** or **spanish** to indicate the language for which to score the predictions file.

The scorer invokes the format checker for the task to verify the output is properly shaped.
It also handles checking if the provided predictions file contains all tweets from the gold one.


## Baselines

The [baselines](./baselines) module currently contains a majority, random and a simple n-gram baseline for subtask 1A and 1B. Additionally, an unoptimized simple baseline of SVM over concatenated Image and Text features is added for Subtask 1A.

**Baseline Results for Task 1 subtasks on Dev_Test**
|Model|subtask-1A--Arabic|subtask-1A--English|subtask-1B--Arabic|subtask-1B--Spanish|subtask-1B--English|
|:----|:----|:----|:----|:----|:----|
|Random Baseline |0.405|0.283|0.429|0.230|0.306|
|Majority Baseline|0.000|0.000|0.000|0.000|0.000|
|n-gram Baseline|0.491|0.645|0.202|0.511|0.821|
|Multimodal:<br/>ResNet+BERT SVM|0.416|0.442|NA|NA|NA|


To launch the baseline script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

### Subtask 1A
To launch the baseline script run the following:
> python3 baselines/subtask_1a.py --data-dir=<path_to_your_subtask_data> --train-file-name=<name_of_your_training_file> --test-file-name=<name_of_your_testing_file> --test-split=<test_split_name> --lang=<language_of_the_subtask><br/>
```
python3 baselines/subtask_1a.py --data-dir=data/CT23_1A_checkworthy_multimodal_english_v1/ --train-file-name=CT23_1A_checkworthy_multimodal_english_train.jsonl --test-file-name=CT23_1A_checkworthy_multimodal_english_dev_test.jsonl --test-split=dev_test --lang=english
```

### Subtask 1B

To launch the baseline script run the following:
> python3 baselines/subtask_1b.py --train-file-path=<path_to_your_training_data> --dev-file-path=<path_to_your_test_data_to_be_evaluated> --lang=<language_of_the_subtask><br/>
```
python3 baselines/subtask_1b.py --train-file-path=data/CT23_1B_checkworthy_arabic/CT23_1B_checkworthy_arabic_train.tsv --dev-file-path=data/CT23_1B_checkworthy_arabic/CT23_1B_checkworthy_arabic_dev.tsv -l arabic
```


All baselines will be trained on the training dataset and the performance of the model is evaluated on the dev_test.

<!-- ## Submission guidelines
Please follow the submission guidelines discussed here: https://sites.google.com/view/clef2022-checkthat/tasks/task-1-identifying-relevant-claims-in-tweets?#h.sn4sm5zguq98.
 -->
## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2023/task1/

<!-- Contact:   clef-factcheck@googlegroups.com -->
