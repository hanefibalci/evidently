from evidently.features.generated_features import FeatureDescriptor
from evidently.features.generated_features import GeneratedFeature
from evidently.features import trigger_words_present_feature


class TriggerWordsPresent(FeatureDescriptor):
    def __init__(self, words_list=(), lemmatisize=True):
        self.words_list = words_list
        self.lemmatisize = lemmatisize

    def feature(self, column_name: str) -> GeneratedFeature:
        return trigger_words_present_feature.TriggerWordsPresent(column_name, self.words_list, self.lemmatisize)

    def for_column(self, column_name: str):
        return trigger_words_present_feature.TriggerWordsPresent(
            column_name,
            self.words_list,
            self.lemmatisize,
        ).feature_name()
