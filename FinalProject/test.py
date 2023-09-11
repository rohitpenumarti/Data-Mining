import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from tqdm import tqdm

if __name__=="__main__":
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    test_df_with_labels = pd.read_csv('y_test.csv')
    final_test_df = pd.merge(test_df, test_df_with_labels, on=['user_id', 'business_id'])

    X_train = train_df[['rounded_business_rating', 'latitude', 'longitude', 'average_user_rating', 'num_user_reviews', 'avg_user_compliments', 
    'reviews_per_year', 'review_count', 'price_range', 'bike_parking', 'business_accepts_credit_cards', 'good_for_kids', 'has_TV', 
    'outdoor_seating', 'restaurants_delivery', 'restaurants_good_for_groups', 'restaurants_reservations', 'restaurants_take_out', 'garage', 
    'street', 'validated', 'lot', 'valet', 'city_state_le', 'name_le', 'max_user_likes', 'total_user_likes', 'avg_user_likes', 
    'max_business_likes', 'total_business_likes', 'avg_business_likes', 'max_ub_pair_likes', 'total_ub_pair_likes', 'avg_ub_pair_likes']]
    y_train = train_df['rating']

    X_test = final_test_df[['rounded_business_rating', 'latitude', 'longitude', 'average_user_rating', 'num_user_reviews', 'avg_user_compliments', 
    'reviews_per_year', 'review_count', 'price_range', 'bike_parking', 'business_accepts_credit_cards', 'good_for_kids', 'has_TV', 
    'outdoor_seating', 'restaurants_delivery', 'restaurants_good_for_groups', 'restaurants_reservations', 'restaurants_take_out', 'garage', 
    'street', 'validated', 'lot', 'valet', 'city_state_le', 'name_le', 'max_user_likes', 'total_user_likes', 'avg_user_likes', 
    'max_business_likes', 'total_business_likes', 'avg_business_likes', 'max_ub_pair_likes', 'total_ub_pair_likes', 'avg_ub_pair_likes']]
    y_test = final_test_df['rating']

    n = len(list(X_train.keys()))-1
    pbar = tqdm(total=n)
    model_performances_with_features = {}
    while n > 0:
        test_xgb = XGBRegressor()
        rfe = RFE(test_xgb, n_features_to_select=n)

        rfe.fit(X_train, y_train)
        features_supp = rfe.support_
        features_out = [list(X_train.keys())[i] for i, supp in enumerate(features_supp) if supp == True]

        test_xgb.fit(X_train[features_out], y_train)
        y_pred_train_rfe = list(test_xgb.predict(X_train[features_out]))
        train_rmse_rfe = mean_squared_error(list(y_train), y_pred_train_rfe, squared=False)

        y_pred_test_rfe = list(test_xgb.predict(X_test[features_out]))
        test_rmse_rfe = mean_squared_error(list(y_test), y_pred_test_rfe, squared=False)
        print(f'##### {n} #####\nTrain RMSE: {train_rmse_rfe}; Test RMSE: {test_rmse_rfe}\n')
        n -= 1
        model_performances_with_features[(train_rmse_rfe, test_rmse_rfe)] = features_out
        pbar.update(1)

    pbar.close()
    best_model = min(model_performances_with_features.items(), key=lambda x: x[0][1])
    print(f'\nBest Model:\nTrain RMSE: {best_model[0][0]}; Test RMSE: {best_model[0][1]}\nBest Features: {best_model[1]}')