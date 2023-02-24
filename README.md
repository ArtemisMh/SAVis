# SAVis: A Learning Analytics Dashboard with Interactive Visualization and Machine Learning
## produced by Zeynab (Artemis) Mohseni, Spring 2020
## Last update: Feb. 2023
---------------------------------------

# What is SAVis:
Student Activity Visualization (SAVis), is a new Learning Analytics Dashboard (LAD) using interactive visualization and Machine Learning (ML) which enables teachers to explore students' learning and activities by interacting with various visualizations of data. SAVis allows to compare groups of students as well as individuals on a different number of categories. Also, the teacher can choose which features to focus on while using SAVis and this allows for greater impact on educational issues rather than technical.


# Short summary about Dataset
Imbalanced dataset typically refers to classification problems where the classes are not represented equally. In this LAD, an imbalanced dataset contained more than 2.3 million samples including the learning behaviors of 6,423 students who used an online study tool with 1,445 educational topics for the period of one year is used. Each educational topic and every student have a unique ID. Each sample of the dataset describes some information including student ID, question type, resource name, resource type, topic ID, student answer, answer duration and the month, the day and the hour that each student answers a question. 

In order to decrease the loading time in the LAD, 10,000 random samples were selected which still allow us to answer all the below research questions. 

# 22 Pedagogical Questions:
we came up with 22 possible pedagogical questions that teachers might want to answer from the data:

* PQ1. How many correct and incorrect answers are there in general? 
* PQ2. What are the maximum and minimum time students spend to answer a question? 
* PQ3. What is the accuracy of the ML algorithm to classify the students to the right university? 
* PQ4. How many correct and incorrect answers are there for each student? 
* PQ5. How many correct and incorrect answers are there for every topic? 
* PQ6. How many correct and incorrect answers are there per month? PQ7. How many correct and incorrect answers are there per day?
* PQ8. How many correct and incorrect answers are there per hour? PQ9. What are the percentages of correct and incorrect answers in general? 
* PQ10. How many correct and incorrect answers are there for different time categories? 
* PQ11. How many correct and incorrect answers are there for every student category? 
* PQ12. How many correct and incorrect answers are there for every topic category? 
* PQ13. How many correct and incorrect answers are there for every question type? 
* PQ14. How many correct and incorrect answers are there for every month of year/day of month/hour of day? 
* PQ15. What hour of the day are students more active?
* PQ16. What day of month are students more active? 
* PQ17. What month of year are students more active? 
* PQ18. Which topic ID has the highest number of correct answers? PQ19. What are the top 10 topic IDs in which students are more active? 
* PQ20. What is the number of correct answers for top 10 topic IDs?
* PQ21. Which question type is the most common? 
* PQ22. What is the percentage of students' activities in different months of the year?

# Publication about SAVis:
https://ceur-ws.org/Vol-2985/paper2.pdf

# Requirements
Python 3.9.7

# How to run this tool?
Create a folder called SAVis in the root folder of your computer by opening the terminal or command prompt. All the files in this repository should be downloaded and placed in the SAVis folder.

```
mkdir SAVis
cd SAVis
```

Install all required packages by running:

```
pip install -r requirements.txt
```

Run this app locally with:

```
python app.py
```

# Screenshot
![GitHub Logo](/Dashboard.png)
