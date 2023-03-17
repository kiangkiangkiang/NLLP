import numpy as np
import os
from paddlenlp import Taskflow
from label_studio_ml.model import LabelStudioMLBase

PARSED_LABEL_CONFIG = {'label': {
    'type': 'Labels',
    'to_name': ['text'],
    'inputs': [{
        'type': 'Text',
        'value': 'text'
    }],
    'labels': ['原告', '非財產上之損害的求償精神慰撫金', '法院得心證判斷之適當裁定精神慰撫金', '其他精神慰撫金'],
    'labels_attrs': {
        '原告': {
            'value': '原告',
            'background': '#44ff00'
        },
        '非財產上之損害的求償精神慰撫金': {
            'value': '非財產上之損害的求償精神慰撫金',
            'background': '#0062ff'
        },
        '法院得心證判斷之適當裁定精神慰撫金': {
            'value': '法院得心證判斷之適當裁定精神慰撫金',
            'background': '#9e06e5'
        },
        '其他精神慰撫金': {
            'value': '其他精神慰撫金',
            'background': '#ff0000'
        }
    }
}
}
LABEL_SCHEMA = ['非財產上之損害的求償精神慰撫金', '法院得心證判斷之適當裁定精神慰撫金',
                {'原告': '非財產上之損害的求償精神慰撫金',
                 '原告': '法院得心證判斷之適當裁定精神慰撫金'}]
DEL_REPEAT_ENTITY = True
PRECISION = 'fp16' #GPU for fp16
MODEL_PATH = "../NLLP/experiment/PaddleNLP_UIE/data_v3/checkpoint/model_best"



class MyUIEPredictModel(LabelStudioMLBase):
    def __init__(self, **kwargs) -> None:
        print('======== in init ========')
        super(MyUIEPredictModel, self).__init__(**kwargs)
        self.parsed_label_config = PARSED_LABEL_CONFIG
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        # print(self.from_name)
        # print(self.info)
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']
        self.labels = list(self.info['labels'])
        self.task_ids = 1
        self.total_entity_id = 0
        self.model = Taskflow("information_extraction", schema=LABEL_SCHEMA,
                                task_path=MODEL_PATH,
                                precision=PRECISION)
        #print(self.labels)

        # task_path='./checkpoint/model_best'
    
    def _del_repeat_entity(self, result):
        start_id_mapping = {}
        index = 0
        while index < len(result) - 1:
            if result[index]['value']['start'] == result[index + 1]['value']['start'] :
                if result[index]['value']['score'] > result[index + 1]['value']['score']:
                    result.pop(index + 1)
                else:
                    result.pop(index)
            else:
                start_id_mapping[result[index]['value']['start']] = result[index]['id']
                index += 1
        return result, start_id_mapping

    def _add_relation(self, result, relations, start_id_mapping):

        for relation in relations:
            people_start = relation['start']
            for relation_type in relation['relations']:
                money_start = relation['relations'][relation_type][0]['start']
                if start_id_mapping.get(people_start) and start_id_mapping.get(money_start):
                    result.append({
                        'from_id': start_id_mapping[people_start],
                        'to_id': start_id_mapping[money_start],
                        "type": "relation",
                        "direction": "right",
                        "labels": [
                            relation_type
                        ]
                    })
        return result
        


    def predict(self, tasks, **kwargs):
        print('======== in predict ========')
        self.model = Taskflow("information_extraction", schema=LABEL_SCHEMA,
                        task_path=MODEL_PATH,
                        precision=PRECISION)

        from_name = self.from_name
        to_name = self.to_name
        model = self.model
        predictions = []
        #print("tasks: ", tasks)
        for task in tasks:
            print("------------------------------------")
            print("Predict task number: ", self.task_ids)
            #print("Predict task:", task)
            text = task['data'][self.value]
            uie = model(text)[0]
            #print(uie)
            result = []
            scores = []
            relations = []
            start_id_mapping = {}

            for key in uie:
                #print("key: ", key)
                for item in uie[key]:
                    print('item: ', item)
                    result.append({
                        'value': {
                            'start': item['start'],
                            'end': item['end'],
                            'score': item['probability'],
                            'text': item['text'],
                            'labels': [key]
                        },
                        'from_name': from_name, 
                        'to_name': to_name,
                        'type': 'labels',
                        'id': str(self.total_entity_id)
                    })
                    self.total_entity_id += 1
                    if item.get('relations') is not None:
                        relations.append(item)
                    scores.append(item['probability'])
                        
            result = sorted(result, key=lambda x: x['value']['start'])

            print("===== result =======")
            print(result)
            if DEL_REPEAT_ENTITY:
                result, start_id_mapping = self._del_repeat_entity(result)
            result = self._add_relation(result, relations, start_id_mapping)

            mean_score = np.mean(scores) if len(scores) else 0
            predictions.append({
                'result': result,
                'score': float(mean_score),
                'model_version': 'uie_new'
            })
            print("result: ", result)
            self.task_ids += 1
        return predictions

    def fit(self, annotations, workdir=None, **kwargs):
        #train
        print('======== in fit ========')
        print("annotations: ", annotations)
        return
