#Dashboard by Artemis, Spring 2020
/* 
Short summary
In this dashboard an imbalanced dataset including the learning behaviors of 6,423 students who used an online study tool with different educational topics for the period of one year is used. 
In this work, after preprocessing and cleaning data, we created a csv file with 1,322,097 samples including 6,423 learning vectors (one per student) in the dataset, each with 1,445 topics.  In order to decrease the loading time in the dashboard, 10,000 samples are selected randomly. 
Since lots of students didn't use the online tool in the appropriate way, there are many unusable data. Proposing a good plot without doing data filtering in uncleaned dataset is not achievable. As a result, preprocessing data and data cleaning are really important.
Preprocessing and cleaning data
The topic ID in this dataset starts from 665. Also, for the topic IDs from 3600-6182 the number of correct and incorrect answers are not impressive. Therefore, a filtering for the topic IDs between 650 and 3500 is applied.
The student answer duration in the dataset is between 0 and 25941 minutes (432 hours). This value shows that lots of students start to answering a question and they never finished it. As a result, there are lots of unusable data that should be dropped. So, a filtering for the student answer time between 0 and 20 minutes is considered.
Goal of developing a dashboard
The focus of this research is to create a dashboard to interact with and analyze the data, deliver the most relevant information to the teachers and help them to understand the data, and make faster and accurate decisions. Before developing a dashboard, we should clean, preprocess and then analyze the data. Also, defining research questions is an important step to find the road map and speed up the process of creating a dashboard. 
In order to analyze the data T-test is used. T-test which usually comes from histogram is a type of inferential statistic used to determine if there is a significant difference between the means of two groups of numerical or quantitative data, which may be related in certain features. this test is not good for big data. In order to calculate the P-value in T-test which is achievable for N<5000, 100 random samples are chosen. P-value for T-test should be less than alpha level to reject the null hypothesis (H0). Research questions and a null hypothesis are as follow:
RQ1.	How many correct/incorrect answers are there in general?
RQ2.	How many correct/incorrect answers are there in different time categories?
RQ3.	How many correct/incorrect answers are there in each student category?
RQ4.	How many correct/incorrect answers are there in each topic category?
RQ5.	How many correct/incorrect answers are there in each question type?
RQ6.	How many correct/incorrect answers are there in each month of year/day of month/hour of day?
RQ7.	Which topic has the highest correct answers? 
RQ8.	What are the top 10 topics that students are more interested? 
RQ9.	Which question type is more interesting for students to answer it? 
RQ10.	What are the number of correct answers for top 10 topics?
RQ11.	What is the maximum/minimum time students spend to answer a question?
RQ12.	What hour of day students are more active?
RQ13.	What day of month students are more active?
RQ14.	What month of year students are more active?

H0.      Expected value in null hypothesis for average answer duration is  minute. 
Proposed Dashboard
A dashboard which provides a central location to monitor and analyze the data is the most efficient way to track multiple data sources. Real-time monitoring reduces the hours of analyzing and long line of communication. 
The main aim of the proposed dashboard is to provide an interactive platform for teachers so they can compare and explore students’ performance by considering different features such as answer duration, topic IDs, student IDs, question type, monthly, daily and hourly activities.
This dashboard also allows to pick two special categories belong to each mentioned feature at a time and compare them by different features. Some interactive techniques such as brushing, zoom, pan and filter are supported in the dashboard. In brushing technique, a subset of elements is selected in one view and this selection is updated and maintained in other views.
Fig. 1 shows the overview of the dashboard. We frequently use levels of increasing detail from (a) key metric to (b) context around the metric to (c) detail for the metric. 
 
Fig. 1. Proposed dashboard
Key metrics includes the number of students, topic IDs, correct and incorrect answers, minimum and maximum answer time and the total number of samples. By choosing each feature from the dropdown component in the top right of the dashboard (Fig. 2), the histogram and its strip plot in the top left of the dashboard are updated. 
 
Fig. 2. Dropdown component
By selecting some elements in the strip plot in the top left of the dashboard, key metrics, other plots and histograms in different tabs will be updated according to the selection. The advantage of this dashboard is that every visualization is interactive, separately. By selecting a part of each visualization, we can zoom the plot and see more details. Also, by clicking on the hued small objects (square or circle) in the right part of each visualization, we can filter the plot according to correct/incorrect answer, month of the activity and question type. 
First histogram in the top right of the dashboard shows the number of correct and incorrect answer for different time categories. In order to see the number of correct/incorrect answers for each category of student and topic, we should choose student ID and Topic ID, respectively, from the dropdown component and select the intended category. By choosing the question type from the dropdown component, we can see the number of correct/incorrect answers in each question type. Also, the number of correct/incorrect answers in each month, day and hour is achievable by choosing monthly activity, daily activity and hourly activity, respectively from the dropdown. Also, from this plot we can see what month of year, what day of month and what hour of day the students are more active.
There are three tabs called “answer duration dashboard”, “topic dashboard” and “activity dashboard” including three sub dashboards. The histogram in the left part of first tab (Fig. 3 (a)) shows the distribution of answer duration. As can be seen, the mean and standard deviation are 0.94 and 1.79. Since the calculated p-value in most of the 100 random samples is less than 0.05 so it is significant and we reject the null hypothesis. The average time for answering different topic questions is less than 1 minute. The scatter plot in the right part of this tab shows students’ monthly activity according to different topic IDs by considering the number of correct and incorrect answer, answer duration and daily activity.
Two plots in left part of Fig. 3 (b) (second tab) show the top 10 topics that students are more interested, the topic that has the highest correct answers, the number of correct answers for top 10 topics. Distribution of topic IDs for student answers. In these two histograms, we can find the number of students’ answers and students’ correct answers for different topic categories.


Last tab (Fig. 3 (c)) illustrates the distribution of students’ monthly activity in different topics, the percentage of the students’ activity per month and the percentages of the students’ activity in different question types.
In the proposed dashboard by using different visualization techniques and applying them on various categorical and numerical data we answered all the mentioned research questions and the null hypothesis. 
*/
