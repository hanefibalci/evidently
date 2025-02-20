from evidently.features import trigger_words_presence_feature
from evidently.features.generated_features import FeatureDescriptor
from evidently.features.generated_features import GeneratedFeature


class TriggerWordsPresence(FeatureDescriptor):
    def __init__(self, words_list=(), lemmatize=True):
        self.words_list = words_list
        self.lemmatize = lemmatize

    def feature(self, column_name: str) -> GeneratedFeature:
        return trigger_words_presence_feature.TriggerWordsPresent(column_name, self.words_list, self.lemmatize)

    def for_column(self, column_name: str):
        return trigger_words_presence_feature.TriggerWordsPresent(
            column_name,
            self.words_list,
            self.lemmatize,
        ).feature_name()
