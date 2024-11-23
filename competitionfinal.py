'''
Firstly, I add all the features of business.json and user.json to the feature vector, and I also extract features from tip.json, photo.json, checkin.json by adding the tips for each business, counting the number of photos for each business and also add up the counts in the checkin,json for each business.
Then, I used Grid Search method to tune the weight of item based method and model based method, then I found out that only use the model-based method performs better and showing best RSME. So I trained the model using only the model-based method which used XGBoost, which lowered the RSME to be less than 0.98.


Error Distribution:                                                                                                                                                        
>=0 and <1: 102305                                                                                                                                                         
>=1 and <2: 32746                                                                                                                                                          
>=2 and <3: 6146                                                                                                                                                           
>=3 and <4: 846                                                                                                                                                            
>=4: 1                                                                                                                                                                     
RMSE:                                                                                                                                                                      
0.9794126177670479   

Execution Time:                                                                                                                                                            
783.6521620750427s
'''




from pyspark import SparkContext
import sys
import time
import csv
import json
from math import sqrt
import math
from statistics import mean
from itertools import combinations
import numpy as np
from xgboost import XGBRegressor


def item_based(train_rdd, test_rdd):
    # Prepare data structures
    bus_user_rate_rdd = train_rdd.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict)
    bus_user_rate_dic = {bus: user_rate for bus, user_rate in bus_user_rate_rdd.collect()}
    
    user_bus_rdd = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set)
    user_bus_dic = user_bus_rdd.collectAsMap()
    
    global_avg = train_rdd.map(lambda x: float(x[2])).mean()

    # Define Pearson correlation function
    def pearson_correlation(bus_ratings1, bus_ratings2, co_rated_threshold=10):
        common_users = set(bus_ratings1.keys()).intersection(bus_ratings2.keys())
        if len(common_users) < co_rated_threshold:
            return 0

        sum1 = sum(bus_ratings1[u] for u in common_users)
        sum2 = sum(bus_ratings2[u] for u in common_users)
        sum1Sq = sum(pow(bus_ratings1[u], 2) for u in common_users)
        sum2Sq = sum(pow(bus_ratings2[u], 2) for u in common_users)
        pSum = sum(bus_ratings1[u] * bus_ratings2[u] for u in common_users)

        num = pSum - (sum1 * sum2 / len(common_users))
        den = sqrt((sum1Sq - pow(sum1, 2) / len(common_users)) * (sum2Sq - pow(sum2, 2) / len(common_users)))
        if den == 0:
            return 0
        return num / den

    # Define the prediction function using Pearson correlation
    def predict(user_id, business_id, bus_user_rate_dic, user_bus_dic, global_avg, top_n):
        if user_id not in user_bus_dic or business_id not in bus_user_rate_dic:
            return global_avg  # Return global average if no data

        user_rated_businesses = user_bus_dic.get(user_id, set())
        co_rated_similarities = []
        for co_rated_business in user_rated_businesses:
            if co_rated_business in bus_user_rate_dic:
                similarity = pearson_correlation(
                    bus_user_rate_dic[business_id],
                    bus_user_rate_dic[co_rated_business]
                )
                if similarity > 0:
                    co_rated_similarities.append((similarity, bus_user_rate_dic[co_rated_business][user_id]))

        if not co_rated_similarities:
            return global_avg  # fallback to business or user average

        co_rated_similarities.sort(key=lambda x: -x[0])
        top_similarities = co_rated_similarities[:top_n]

        weighted_sum = sum(sim * rate for sim, rate in top_similarities)
        normalizer = sum(abs(sim) for sim, _ in top_similarities)

        if normalizer == 0:
            return global_avg  # Avoid division by zero

        prediction = weighted_sum / normalizer
        return prediction

    # Predict ratings for test dataset
    predict_rdd = test_rdd.map(lambda x: (x[0], x[1], predict(x[0], x[1], bus_user_rate_dic, user_bus_dic, global_avg, 30)))
    return predict_rdd.collect()


def parse_attributes(attributes):
    if attributes is None:
        return {
            'BikeParking': False,
            'BusinessAcceptsCreditCards': False,
            'GoodForKids': False,
            'HasTV': False,
            'NoiseLevel': 1,  # Default to 'average' → 1
            'OutdoorSeating': False,
            'RestaurantsAttire': 0,  # Numeric: 'casual' → 0, 'formal' → 1
            'RestaurantsDelivery': False,
            'RestaurantsGoodForGroups': False,
            'RestaurantsPriceRange2': 2,
            'RestaurantsReservations': False,
            'RestaurantsTakeOut': False
        }
    
    noise_level_map = {
        'quiet': 0,
        'average': 1,
        'loud': 2,
        'very_loud': 3
    }

    return {
        'BikeParking': attributes.get('BikeParking', 'False') == 'True',
        'BusinessAcceptsCreditCards': attributes.get('BusinessAcceptsCreditCards', 'False') == 'True',
        'GoodForKids': attributes.get('GoodForKids', 'False') == 'True',
        'HasTV': attributes.get('HasTV', 'False') == 'True',
        'NoiseLevel': noise_level_map.get(attributes.get('NoiseLevel', 'average'), 1),
        'OutdoorSeating': attributes.get('OutdoorSeating', 'False') == 'True',
        'RestaurantsAttire': 0 if attributes.get('RestaurantsAttire', 'casual') == 'casual' else 1,
        'RestaurantsDelivery': attributes.get('RestaurantsDelivery', 'False') == 'True',
        'RestaurantsGoodForGroups': attributes.get('RestaurantsGoodForGroups', 'False') == 'True',
        'RestaurantsPriceRange2': int(attributes.get('RestaurantsPriceRange2', '2')),
        'RestaurantsReservations': attributes.get('RestaurantsReservations', 'False') == 'True',
        'RestaurantsTakeOut': attributes.get('RestaurantsTakeOut', 'False') == 'True'
    }



def extract_business_features(business):
    # Extend business_fields_dict with new attributes and categories
    attributes = parse_attributes(business.get('attributes', {}))
    features = (
        business['stars'],
        business['review_count'],
        business['is_open'],
        attributes['BikeParking'],
        attributes['BusinessAcceptsCreditCards'],
        attributes['GoodForKids'],
        attributes['HasTV'],
        attributes['NoiseLevel'],
        attributes['OutdoorSeating'],
        attributes['RestaurantsAttire'],
        attributes['RestaurantsDelivery'],
        attributes['RestaurantsGoodForGroups'],
        attributes['RestaurantsPriceRange2'],
        attributes['RestaurantsReservations'],
        attributes['RestaurantsTakeOut']
    )
    return (business['business_id'], features)


def model_based(train_rdd, val_rdd):
    tip_path = folder_path + '/tip.json'
    #tip_path = 'tip.json'
    tips_rdd = sc.textFile(tip_path).map(lambda line: json.loads(line)) 
    business_likes_dict = tips_rdd.map(lambda x: (x['business_id'], x['likes'])) \
                             .reduceByKey(lambda a, b: a + b).collectAsMap()
        
    photo_path = folder_path + '/photo.json'  # Adjust path as necessary
    photos_rdd = sc.textFile(photo_path).map(lambda line: json.loads(line))
    business_photo_count = photos_rdd.map(lambda x: (x['business_id'], 1)) \
                                 .reduceByKey(lambda a, b: a + b).collectAsMap()
    checkin_path = folder_path + '/checkin.json'
    checkins_rdd = sc.textFile(checkin_path).map(lambda line: json.loads(line))
    business_checkin_count = checkins_rdd.map(lambda x: (x['business_id'], sum(x['time'].values()))) \
                                     .reduceByKey(lambda a, b: a + b).collectAsMap()
        
    
    #business_path = 'business.json'
    business_rdd = sc.textFile(folder_path + '/business.json').map(lambda line: json.loads(line))
    business_fields_dict = business_rdd.map(extract_business_features).collectAsMap()

    user_path = folder_path + '/user.json'
    #user_path = 'user.json'
    user_rdd = sc.textFile(user_path)
    user_rdd = user_rdd.map(lambda x: json.loads(x))
    user_fields_dict = user_rdd.map(lambda user: (
        user['user_id'],
        (user['review_count'],
        user['average_stars'],
        user['useful'],
        user['funny'],
        user['cool'],
        user['fans'],
        user['compliment_hot'],
        user['compliment_more'],
        user['compliment_profile'],
        user['compliment_cute'],
        user['compliment_list'],
        user['compliment_note'],
        user['compliment_plain'],
        user['compliment_cool'],
        user['compliment_funny'],
        user['compliment_writer'],
        user['compliment_photos']
    )
    )).collectAsMap()
    

    features_list = []
    ratings_list = []

    # Iterate through each enhanced record to append business, user features, and average star ratings
    for record in train_rdd.collect():
        user_id, business_id, rating = record
        ratings_list.append(float(rating))
        feature_vector = []

        # Existing business features
        business_features = business_fields_dict.get(business_id, [None] * 6)
        feature_vector.extend(business_features)

        # Existing user features
        user_features = user_fields_dict.get(user_id, [None] * 17)
        feature_vector.extend(user_features)

        features_list.append(feature_vector)

    x_train = np.array(features_list, dtype='float32')
    y_train = np.array(ratings_list, dtype='float32')

    features_list_val = []

    for record in val_rdd.collect():
        user_id, business_id,rating = record
        feature_vector_val = []
        
        business_features = business_fields_dict.get(business_id, [None, None]) 
        feature_vector_val.extend(business_features)
        
        user_features = user_fields_dict.get(user_id, [None, None])
        feature_vector_val.extend(user_features)
        
        features_list_val.append(feature_vector_val)

    x_val = np.array(features_list_val)
    xgb = XGBRegressor(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=1000,
        booster='gbtree',
        gamma=0,
        min_child_weight=1,
        subsample=1,
        colsample_bytree=1,
        reg_alpha=0.01,
        reg_lambda=0,
        random_state=0
    )
    xgb.fit(x_train, y_train)
    y_pred = xgb.predict(x_val)

    val_ids = [(record[0], record[1]) for record in val_rdd.collect()]
    
    predictions_with_ids = []
    for ids, pred in zip(val_ids, y_pred):
        predictions_with_ids.append((ids[1], ids[0], pred))
    return predictions_with_ids

'''
def grid_search(combined_rdd, test_rdd, step=0.01):
    best_rmse = float('inf')
    best_weights = (0, 0)
    for i in np.arange(0, 1+step, step):
        weight_item_based = i
        weight_model_based = 1 - i
        
        # Calculate combined predictions
        weighted_predictions = combined_rdd.map(lambda x: (
            x[0],  # key
            x[1][0] * weight_item_based + x[1][1] * weight_model_based  # weighted sum of ratings
        ))
        
        # Join with test ratings to calculate error
        test_keyed = test_rdd.map(lambda x: ((x[0], x[1]), x[2]))  # assuming test_rdd is in the same format
        predictions_and_actuals = weighted_predictions.join(test_keyed)
        
        mse = predictions_and_actuals.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean()
        rmse = sqrt(mse)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (weight_item_based, weight_model_based)
        #print(f"Weight for Item-based: {weight_item_based}, Model-based: {weight_model_based}, RMSE: {rmse}")
    return best_weights, best_rmse
'''

start_time = time.time()
sc = SparkContext('local[*]', 'competition')
sc.setLogLevel('WARN')

folder_path = sys.argv[1]
val_path = sys.argv[2]
output_file_path = sys.argv[3]

'''
val_path = 'yelp_val.csv'
output_file_path = 'output.csv'
'''

train_rdd = sc.textFile(folder_path + '/yelp_train.csv')
#train_rdd = sc.textFile('yelp_train.csv')
train_header = train_rdd.first()
train_rdd = train_rdd.filter(lambda line: line != train_header).map(lambda line: line.split(','))

val_rdd = sc.textFile(val_path)
val_header = val_rdd.first()
val_rdd = val_rdd.filter(lambda line: line != val_header).map(lambda line: line.split(','))
test_y = val_rdd.map(lambda x: float(x[2]))

item_based_results = item_based(train_rdd, val_rdd)
model_based_results = model_based(train_rdd, val_rdd)

#jia grid search
#item_based_rdd = sc.parallelize(item_based_results)
#model_based_rdd = sc.parallelize(model_based_results)
#item_based_keyed = item_based_rdd.map(lambda x: ((x[0], x[1]), x[2]))
#model_based_keyed = model_based_rdd.map(lambda x: ((x[0], x[1]), x[2]))
#combined_rdd = item_based_keyed.join(model_based_keyed)

#best_weights, best_rmse = grid_search(combined_rdd, val_rdd)
#print(best_weights)
results = []
for i in range(len(item_based_results)):
    user_id, business_id, item_rating = item_based_results[i]
    _, _, model_rating = model_based_results[i] 
    final_score = model_rating
    results.append((user_id, business_id, final_score))
results_rdd = sc.parallelize(results)
predict_y = results_rdd.map(lambda x: x[2])
predict_y_indexed = predict_y.zipWithIndex().map(lambda x: (x[1], x[0]))  # (index, prediction)
test_y_indexed = test_y.zipWithIndex().map(lambda x: (x[1], x[0]))  # (index, actual)
joined_rdd = predict_y_indexed.join(test_y_indexed)
ab_diff = joined_rdd.map(lambda x: abs(x[1][0] - x[1][1]))

error_buckets = ab_diff.histogram([0, 1, 2, 3, 4, 5])[1]  # Using histogram to bin the differences
error_labels = [
    ">=0 and <1",
    ">=1 and <2",
    ">=2 and <3",
    ">=3 and <4",
    ">=4"
]


with open(output_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['user_id', ' business_id', ' prediction']) 

    for user_id, business_id, final_score in results:
        writer.writerow([user_id, business_id, final_score])
        
'''
print("Error Distribution:")
for label, count in zip(error_labels, error_buckets):
    print(f"{label}: {count}")

squared_errors = joined_rdd.map(lambda xy: (xy[1][0] - xy[1][1])**2)
mse = squared_errors.mean()
rmse = sqrt(mse)

print("RMSE:")
print(rmse)

print('Execution Time: ')
print(str(time.time() - start_time) + 's')
'''