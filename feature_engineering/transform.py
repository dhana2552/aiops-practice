import tensorflow as tf
import tensorflow_transform as tft


columns = ['pickup_community_area', 'fare', 'trip_start_month', 'trip_start_hour',
       'trip_start_day', 'trip_start_timestamp', 'pickup_latitude',
       'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
       'trip_miles', 'pickup_census_tract', 'dropoff_census_tract',
       'payment_type', 'company', 'trip_seconds', 'dropoff_community_area',
       'tips']

_DENSE_FLOAT_FEATURE_KEYS = ['fare', 'trip_miles', 'trip_seconds']
_BUCKET_FEATURE_KEYS = ['pickup_latitude',
       'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
_VOCAB_FEATURE_KEYS = ['payment_type', 'company']
_LABEL_KEY = 'tips'
_FARE_KEY = 'fare'
_VOCAB_SIZE = 10000
_OOV_SIZE = 10
_BUCKET_SIZE = 10

def _transformed_name(key):
  return key + '_xf'

def transformed_names(keys):
  return [_transformed_name(key) for key in keys]

def _fill_in_missing(x):
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]), default_value)

def preprocessing_fn(input):
    output={}
    print(f"Standardization of dense columns: {_DENSE_FLOAT_FEATURE_KEYS}")
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        output[_transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(input[key]))
        
    print(f"Transorformation of categorical columns: {_VOCAB_FEATURE_KEYS}")
    for key in _VOCAB_FEATURE_KEYS:
        output[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(input[key]), top_k = _VOCAB_SIZE, oov_buckets = _OOV_SIZE)
        
    print(f"Bucketization of bucket columns: {_BUCKET_FEATURE_KEYS}")
    for key in _BUCKET_FEATURE_KEYS:
        output[_transformed_name(key)] = tft.bucketize(
            _fill_in_missing(input[key]), _BUCKET_SIZE)
        
    taxi_fare = _fill_in_missing(input[_FARE_KEY])
    tips = _fill_in_missing(input[_LABEL_KEY])
    
    output[_transformed_name(_LABEL_KEY)] = tf.where(
            tf.math.is_nan(taxi_fare),
            tf.cast(tf.zeros_like(taxi_fare), tf.int64),
            tf.cast(tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.10))), tf.int64)
        )
    return output




