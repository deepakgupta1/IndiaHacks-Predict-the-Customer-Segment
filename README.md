# IndiaHacks-Predict-the-Segment
Predict the segment to which a Hotstar customer belongs to based on the watch patterns.

### Problem.
Need a machine learning based solution using which we can learn patterns from customers whose watch patterns are already known. In this competition, the task is to generate predictive models that can best capture the behaviour. 

### Data.
The training dataset consists of data corresponding to 200,000 customers and the test dataset consists of 100,000 customers. Both training and test data is in the form of json dict, where key is masked user ID and value is aggregation of all records corresponding to the user.

### Approach.
* Find the most watched shows, the most watched locations from.
* Make lot of features.
* 5 fold cv.
* xgbclassifier.

### Features:
* count of watched city, genres, titles, days, hours.
* average of watched city, genres, titles, days, hours.
* watched times in top 20 shows from complete data set.
* watched times in top 25 locations from complete data set.
* watched times in top few genres from complete data set and rest clubbed to other_genre.
* watched times in all days and hours.

### Major Libraries used:
* json
* datetime
* xgboost
* sklearn
* pandas
