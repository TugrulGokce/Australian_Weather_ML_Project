from collections import Counter
import streamlit as st
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib


australian_weather = pd.read_csv('dataset/australia_weather.csv')

# Specify target and features then splitting dataset
X = australian_weather.drop(["RainTomorrow"], axis=1)
Y = australian_weather["RainTomorrow"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

target_class_val_cnt = australian_weather.RainTomorrow.value_counts()

st.title('Australian Weather Prediction')

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('kNN', 'XgBoostClassifier', 'Logistic Regression', 'Linear SVC', 'Naive Bayes')
)

sampling_name = st.sidebar.selectbox(
    'Select sampling methods',
    ('Oversampling', 'SMOTE', 'ADASYN', 'Undersampling', '<No Sampling>')
)


def get_classifer(clf_name, samp_name):
    if clf_name == "kNN":
        st.header(f"**Using {clf_name} algorithm**")
        apply_sampling_methods(samp_name)
        if samp_name == 'Oversampling':
            knn_ros = joblib.load("ML_Models_Joblib/knn_ros_model")
            return knn_ros
        elif samp_name == 'SMOTE':
            knn_smote = joblib.load("ML_Models_Joblib/knn_smote_model")
            return knn_smote
        elif samp_name == 'ADASYN':
            knn_adasyn = joblib.load("ML_Models_Joblib/knn_adasyn_model")
            return knn_adasyn
        elif samp_name == 'Undersampling':
            knn_rus = joblib.load("ML_Models_Joblib/knn_rus_model")
            return knn_rus
        else:  # <No Sampling>
            knn_no_samp = joblib.load("ML_Models_Joblib/kNN_model")
            return knn_no_samp

    elif clf_name == 'XgBoostClassifier':
        st.header(f"**Using {clf_name} algorithm**")
        apply_sampling_methods(samp_name)
        if samp_name == 'Oversampling':
            xgb_ros = joblib.load("ML_Models_Joblib/xgb_ros_model")
            return xgb_ros
        elif samp_name == 'SMOTE':
            xgb_smote = joblib.load("ML_Models_Joblib/xgb_smote_model")
            return xgb_smote
        elif samp_name == 'ADASYN':
            xgb_adasyn = joblib.load("ML_Models_Joblib/xgb_adasyn_model")
            return xgb_adasyn
        elif samp_name == 'Undersampling':
            xgb_rus = joblib.load("ML_Models_Joblib/xgb_rus_model")
            return xgb_rus
        else:  # <No Sampling>
            xgb_no_samp = joblib.load("ML_Models_Joblib/xgboost_classifer_model")
            return xgb_no_samp

    elif clf_name == 'Logistic Regression':
        st.header(f"**Using {clf_name} algorithm**")
        apply_sampling_methods(samp_name)
        if samp_name == 'Oversampling':
            logreg_ros = joblib.load("ML_Models_Joblib/logreg_ros_model")
            return logreg_ros
        elif samp_name == 'SMOTE':
            logreg_smote = joblib.load("ML_Models_Joblib/logreg_smote_model")
            return logreg_smote
        elif samp_name == 'ADASYN':
            logreg_adasyn = joblib.load("ML_Models_Joblib/logreg_adasyn_model")
            return logreg_adasyn
        elif samp_name == 'Undersampling':
            logreg_rus = joblib.load("ML_Models_Joblib/logreg_rus_model")
            return logreg_rus
        else:  # <No Sampling>
            logreg_no_samp = joblib.load("ML_Models_Joblib/log_reg_model")
            return logreg_no_samp

    elif clf_name == 'Linear SVC':
        st.header(f"**Using {clf_name} algorithm**")
        apply_sampling_methods(samp_name)
        if samp_name == 'Oversampling':
            linear_svc_ros = joblib.load("ML_Models_Joblib/linear_svc_ros_model")
            return linear_svc_ros
        elif samp_name == 'SMOTE':
            linear_svc_smote = joblib.load("ML_Models_Joblib/linear_svc_smote_model")
            return linear_svc_smote
        elif samp_name == 'ADASYN':
            linear_svc_adasyn = joblib.load("ML_Models_Joblib/linear_svc_adasyn_model")
            return linear_svc_adasyn
        elif samp_name == 'Undersampling':
            linear_svc_rus = joblib.load("ML_Models_Joblib/linear_svc_rus_model")
            return linear_svc_rus
        else:  # <No Sampling>
            svc_no_samp = joblib.load("ML_Models_Joblib/svc_model")
            return svc_no_samp

    else:  # Naive Bayes
        st.header(f"**Using {clf_name} algorithm**")
        apply_sampling_methods(samp_name)
        if samp_name == 'Oversampling':
            nb_ros = joblib.load("ML_Models_Joblib/nb_ros_model")
            return nb_ros
        elif samp_name == 'SMOTE':
            nb_smote = joblib.load("ML_Models_Joblib/nb_smote_model")
            return nb_smote
        elif samp_name == 'ADASYN':
            nb_adasyn = joblib.load("ML_Models_Joblib/nb_adasyn_model")
            return nb_adasyn
        elif samp_name == 'Undersampling':
            nb_rus = joblib.load("ML_Models_Joblib/nb_rus_model")
            return nb_rus
        else:  # <No Sampling>
            nb_no_samp = joblib.load("ML_Models_Joblib/Naive_Bayes_model")
            return nb_no_samp


def apply_sampling_methods(samp_name):
    if samp_name == 'Oversampling':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, Y)
        counter_y_resampler = Counter(y_resampled)
        st.write(f"Original target values distribution : **No = {target_class_val_cnt[0]},"
                 f" Yes = {target_class_val_cnt[1]}**")
        st.write(
            f"New target values distribution : **No = {counter_y_resampler[0]}, Yes = {counter_y_resampler[1]}**")
    elif samp_name == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_smoted, y_smoted = smote.fit_resample(X, Y)
        counter_y_smoted = Counter(y_smoted)
        st.write(f"Original target values distribution : **No = {target_class_val_cnt[0]},"
                 f" Yes = {target_class_val_cnt[1]}**")
        st.write(f"New target values distribution : **No = {counter_y_smoted[0]}, Yes = {counter_y_smoted[1]}**")
    elif samp_name == 'ADASYN':
        adasyn = ADASYN(random_state=42)
        X_adasyn, y_adasyn = adasyn.fit_resample(X, Y)
        counter_y_adasyn = Counter(y_adasyn)
        st.write(f"Original target values distribution : **No = {target_class_val_cnt[0]},"
                 f" Yes = {target_class_val_cnt[1]}**")
        st.write(f"New target values distribution : **No = {counter_y_adasyn[0]}, Yes = {counter_y_adasyn[1]}**")
    elif samp_name == 'Undersampling':  # Undersampling
        randomus = RandomUnderSampler(random_state=42)
        X_under, y_under = randomus.fit_resample(X, Y)
        counter_y_randomus = Counter(y_under)
        st.write(f"Original target values distribution : **No = {target_class_val_cnt[0]},"
                 f" Yes = {target_class_val_cnt[1]}**")
        st.write(
            f"New target values distribution : **No = {counter_y_randomus[0]}, Yes = {counter_y_randomus[1]}**")
    else:  # <No Sampling>
        st.write(f"Original target values distribution : **No = {target_class_val_cnt[0]},"
                 f" Yes = {target_class_val_cnt[1]}**")


clf_model = get_classifer(classifier_name, sampling_name)

fig, ax = plt.subplots()
heat_map_cm = sns.heatmap(confusion_matrix(y_test, clf_model.predict(X_test)), cmap="Greens", annot=True,
                          annot_kws={"size": 15}, fmt='d')
plt.xlabel("Predicted", fontdict={"fontsize": 15})
plt.ylabel("Actual", fontdict={"fontsize": 15})
plt.title(f"Australian Weather Dataset Confusion Matrix with {sampling_name}", fontdict={'fontsize': 18}, color="green")
plt.switch_backend('agg')
heat_map_cm.set_xticklabels(heat_map_cm.get_xmajorticklabels(), fontsize=14)
heat_map_cm.set_yticklabels(heat_map_cm.get_ymajorticklabels(), fontsize=14)
st.write(fig)
