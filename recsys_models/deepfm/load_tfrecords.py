import tensorflow as tf

tf.enable_eager_execution()
feature_description = {
    "u_id": tf.FixedLenFeature(dtype=tf.int64, shape=1),
    "i_id": tf.FixedLenFeature(dtype=tf.int64, shape=1),
    "i_channel": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_brand": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_operator": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_activelevel": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_age": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_marriage": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_sex": tf.FixedLenFeature(dtype=tf.string, shape=1, ),
    "u_sex_age": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_sex_marriage": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "u_age_marriage": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "i_hot_news": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "i_is_recommend": tf.FixedLenFeature(dtype=tf.string, shape=1),
    "i_info_exposed_amt": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_info_clicked_amt": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_info_ctr": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_cate_exposed_amt": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_cate_clicked_amt": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_category_ctr": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_ctr_1": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_clicked_amt_1": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_exposed_amt_1": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_ctr_3": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_clicked_amt_3": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_exposed_amt_3": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_ctr_7": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_clicked_amt_7": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_exposed_amt_7": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_ctr_14": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_clicked_amt_14": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_uid_type_exposed_amt_14": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "c_user_flavor": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at1": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at2": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at3": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at4": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "u_activetime_at5": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_mini_img_size": tf.FixedLenFeature(dtype=tf.float32, shape=1),
    "i_comment_count": tf.FixedLenFeature(dtype=tf.float32, shape=1),
}
categoryFeatureNa = '####'


def _parse_example(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features=feature_description)
    return features


datadir = ['part-r-00000']
dataset = tf.data.TFRecordDataset(datadir)
# dataset = dataset.map(_parse_example, num_parallel_calls=8)
# for i in dataset.take(1):
#     print(i)
#
# saved_model_cli run --dir /home/wangrc/Downloads/export/1560231853/ --tag_set serve --signature_def serving_default --input_exprs='examples=[b"\n\x8d\t\n\x16\n\ti_channel\x12\t\n\x07\n\x05meinv\n\x1e\n\x12i_info_exposed_amt\x12\x08\x12\x06\n\x04\x00\x00@B\n\x1e\n\x12i_info_clicked_amt\x12\x08\x12\x06\n\x04\x00\x00\xc0@\n\x16\n\ni_info_ctr\x12\x08\x12\x06\n\x04\x00\x00\x00>\n\x1e\n\x12i_cate_exposed_amt\x12\x08\x12\x06\n\x04`\xca_J\n\x1e\n\x12i_cate_clicked_amt\x12\x08\x12\x06\n\x04\x00r\xefH\n\x1a\n\x0ei_category_ctr\x12\x08\x12\x06\n\x040\xf4\x08>\n\x1c\n\x10c_uid_type_ctr_1\x12\x08\x12\x06\n\x04NG\x08?\n$\n\x18c_uid_type_clicked_amt_1\x12\x08\x12\x06\n\x04\x00\x00\xd6B\n$\n\x18c_uid_type_exposed_amt_1\x12\x08\x12\x06\n\x04\x00\x00IC\n\x1c\n\x10c_uid_type_ctr_3\x12\x08\x12\x06\n\x040\xba\x08?\n$\n\x18c_uid_type_clicked_amt_3\x12\x08\x12\x06\n\x04\x00@\x01D\n$\n\x18c_uid_type_exposed_amt_3\x12\x08\x12\x06\n\x04\x00\x00rD\n\x1c\n\x10c_uid_type_ctr_7\x12\x08\x12\x06\n\x040\xba\x08?\n$\n\x18c_uid_type_clicked_amt_7\x12\x08\x12\x06\n\x04\x00@\x01D\n$\n\x18c_uid_type_exposed_amt_7\x12\x08\x12\x06\n\x04\x00\x00rD\n\x1d\n\x11c_uid_type_ctr_14\x12\x08\x12\x06\n\x040\xba\x08?\n%\n\x19c_uid_type_clicked_amt_14\x12\x08\x12\x06\n\x04\x00@\x01D\n%\n\x19c_uid_type_exposed_amt_14\x12\x08\x12\x06\n\x04\x00\x00rD\n\x19\n\rc_user_flavor\x12\x08\x12\x06\n\x04]\xfe\x83?\n\x16\n\nu_operator\x12\x08\n\x06\n\x04####\n\x1c\n\x10u_activetime_at1\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1c\n\x10u_activetime_at2\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1c\n\x10u_activetime_at3\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1c\n\x10u_activetime_at4\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1c\n\x10u_activetime_at5\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1b\n\x0fi_mini_img_size\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x1b\n\x0fi_comment_count\x12\x08\x12\x06\n\x04\x00\x00\x00\x00\n\x13\n\x07u_brand\x12\x08\n\x06\n\x04####\n\x19\n\ru_activelevel\x12\x08\n\x06\n\x04####\n\x11\n\x05u_age\x12\x08\n\x06\n\x04####\n\x16\n\nu_marriage\x12\x08\n\x06\n\x04####\n\x11\n\x05u_sex\x12\x08\n\x06\n\x04####\n\x15\n\tu_sex_age\x12\x08\n\x06\n\x04####\n\x1a\n\x0eu_sex_marriage\x12\x08\n\x06\n\x04####\n\x1a\n\x0eu_age_marriage\x12\x08\n\x06\n\x04####\n\x16\n\ni_hot_news\x12\x08\n\x06\n\x04####\n\x1a\n\x0ei_is_recommend\x12\x08\n\x06\n\x04####\n\x0f\n\x04u_id\x12\x07\x1a\x05\n\x03\xe9\xde?\n\x0f\n\x04i_id\x12\x07\x1a\x05\n\x03\x94\x89\x1c"]'
